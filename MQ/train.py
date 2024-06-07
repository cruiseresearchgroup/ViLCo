# python imports
import argparse
import os
import time
import datetime
from pprint import pprint
import copy
# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma, validate_loss, valid_one_epoch)
import logging
from eval import valid_performance
from torch.distributed import init_process_group, destroy_process_group
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
    # pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename)
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if int(os.environ["LOCAL_RANK"]) == 0 and (not os.path.exists(ckpt_folder)):
        os.mkdir(ckpt_folder)
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    log_dir = ckpt_folder
    if int(os.environ["LOCAL_RANK"]) == 0 and (not os.path.exists(log_dir)):
        os.makedirs(log_dir)

    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    n_gpu = torch.cuda.device_count()
    # cfg['opt']["learning_rate"] *= torch.cuda.device_count()
    # cfg['loader']['num_workers'] *= torch.cuda.device_count()


    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )


    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    # model = nn.DataParallel(model, device_ids=cfg['devices'])
    gpu_id = int(os.environ["LOCAL_RANK"])
    model = model.to(gpu_id)
    
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)
    if model_ema is not None:
        model_ema = model_ema.to(gpu_id)
    # model_ema = None

    val_db_vars = val_dataset.get_attributes()
    evaluator = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds = val_db_vars['tiou_thresholds']
    )

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                map_location = lambda storage, loc: storage.cuda(gpu_id))
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # also load the optimizer / scheduler if necessary
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{:s}' (epoch {:d}".format(
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
    best_epoch_of_avgmap = -1
    best_avgmap = -10000.0
    best_recall = None
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            n_gpu,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=args.print_freq, logger=logger
        )
        # import ipdb;ipdb.set_trace()
        # val_loss = validate_loss(
        #     val_loader,
        #     model,
        #     epoch,
        #     clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
        #     tb_writer=tb_writer,
        #     print_freq=args.print_freq, logger=logger
        # )
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     save_states = {
        #         'epoch': epoch,
        #         'state_dict': model.state_dict(),
        #         'scheduler': scheduler.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     if model_ema is not None:
        #         save_states['state_dict_ema'] = model_ema.module.state_dict()
        #     save_checkpoint(
        #         save_states,
        #         file_folder=ckpt_folder,
        #         file_name='best_val_loss.pth.tar'.format(epoch)
        #     )

        # ============= infer each epoch ==========
        if int(os.environ["LOCAL_RANK"]) == 0 and (not args.combine_train):
            if epoch < max_epochs // 3:
            # if epoch < 0:
                continue
            logger.info(f"start validate map&recall of epoch {epoch}")
            with torch.no_grad():
                cur_model = copy.deepcopy(model)
                cur_model.load_state_dict(model_ema.module.state_dict())
                mAP, avg_mAP, tiou_thresholds, eval_result = valid_one_epoch(val_loader, cur_model, epoch, 
                                                        evaluator=evaluator, tb_writer=None,
                                                        logger=logger, dataset_name=cfg['dataset_name'], print_freq=100)

            if avg_mAP > best_avgmap:
                best_avgmap = avg_mAP
                best_epoch_of_avgmap = epoch
                best_recall = eval_result
                best_map = mAP
                save_states = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if model_ema is not None:
                    save_states['state_dict_ema'] = model_ema.module.state_dict()
                save_checkpoint(
                    save_states,
                    file_folder=ckpt_folder,
                    file_name='best_performance.pth.tar'.format(epoch)
                )
            if cfg['dataset_name'] == "ego4d":
                tious = [0.1, 0.2, 0.3, 0.4, 0.5]
                recalls = [1, 5]
                recall1x5 = best_recall[4, 0]    
                logger.info(f'Current Best Recall 1@0.5 is : [epoch {best_epoch_of_avgmap}], {recall1x5 * 100: .2f} %')
            logger.info(f'Current Best Average Map is  : [epoch {best_epoch_of_avgmap}], {best_avgmap * 100: .2f} %')
        else:
            if int(os.environ["LOCAL_RANK"]) == 0 and (epoch > max_epochs - 5):
            # if epoch == 11:
                save_states = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                if model_ema is not None:
                    save_states['state_dict_ema'] = model_ema.module.state_dict()
                save_checkpoint(
                    save_states,
                    file_folder=ckpt_folder,
                    file_name='epoch_{:03d}.pth.tar'.format(epoch)
                )
        # ============= infer each epoch ==========

    # save ckpt once in a while
    # save_states = {
    #     'epoch': epoch,
    #     'state_dict': model.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    # }
    # if model_ema is not None:
    #     save_states['state_dict_ema'] = model_ema.module.state_dict()

    # save_checkpoint(
    #     save_states,
    #     file_folder=ckpt_folder,
    #     file_name='epoch_{:03d}.pth.tar'.format(epoch)
    # )

    # wrap up
    tb_writer.close()
    if int(os.environ["LOCAL_RANK"]) == 0:
        destroy_process_group()

    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    parser.add_argument('--combine_train', action='store_true')
    args = parser.parse_args()

    main(args)
