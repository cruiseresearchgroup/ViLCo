# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.utils.data
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, save_checkpoint, make_optimizer, make_scheduler, fix_random_seed, ModelEma)
from libs.utils.train_utils import valid_one_epoch_loss
from libs.utils.model_utils import count_parameters


################################################################################
def main(args):
    """main function that handles training / inference"""

    """1. setup parameters / folders"""
    init_process_group(backend="nccl")

    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")

    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')

    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))

    if int(os.environ["LOCAL_RANK"]) == 0:
        pprint(cfg)
        os.makedirs(ckpt_folder, exist_ok=True)

    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= torch.cuda.device_count()
    # cfg['loader']['num_workers'] *= torch.cuda.device_count()
    print(cfg['opt']["learning_rate"])

    """2. create dataset / dataloader"""

    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, **cfg['loader']
    )

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    if int(os.environ["LOCAL_RANK"]) == 0:
        print(model)
        count_parameters(model)

    # enable model EMA
    # print("Using model EMA ...")
    model_ema = ModelEma(model)

    gpu_id = int(os.environ["LOCAL_RANK"])
    model = model.to(gpu_id)
    # model = DDP(model, device_ids=[gpu_id])

    if model_ema is not None:
        model_ema = model_ema.to(gpu_id)

    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(gpu_id))
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            if args.resume_from_pretrain:
                args.start_epoch = 0
            else:
                args.start_epoch = checkpoint['epoch'] + 1
                try:
                    model_ema.load_state_dict(checkpoint['state_dict_ema'])
                except:
                    pass
                # also load the optimizer / scheduler if necessary
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d})".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """4. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # start training
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    score_writer = open(os.path.join(ckpt_folder, "eval_results.txt"), mode="w", encoding="utf-8")

    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq
        )

        # save ckpt once in a while
        if (
                (epoch == max_epochs - 1) or
                (
                        (args.ckpt_freq > 0) and
                        (epoch % args.ckpt_freq == 0)
                )
        ):
            print("\nStart testing model {:s} ...".format(cfg['model_name']))
            start = time.time()
            losses_tracker = valid_one_epoch_loss(
                val_loader,
                model,
                epoch,
                tb_writer=tb_writer,
                print_freq=args.print_freq / 2
            )
            end = time.time()
            print("All done! Total time: {:0.2f} sec".format(end - start))
            # print("losses_tracker: ", losses_tracker)
            score_str = ""

            for key, value in losses_tracker.items():
                score_str += '\t{:s} {:.2f} ({:.2f})'.format(
                    key, value.val, value.avg
                )

            score_writer.write(score_str)
            score_writer.flush()

        if int(os.environ["LOCAL_RANK"]) == 0:
            save_states = {'epoch': epoch,
                           'state_dict': model.state_dict(),
                           'scheduler': scheduler.state_dict(),
                           'optimizer': optimizer.state_dict(),
                           'state_dict_ema': model_ema.module.state_dict(),
                           }

            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )

    # wrap up
    tb_writer.close()
    if int(os.environ["LOCAL_RANK"]) == 0:
        destroy_process_group()


################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=2, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='./ckpt', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--resume_from_pretrain', default=False, type=bool)
    args = parser.parse_args()
    main(args)
