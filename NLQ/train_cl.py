# python imports
import argparse
import os
import time
import datetime
import pickle
import numpy as np
from pprint import pprint

# torch imports
import torch
import torch.utils.data
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader, QILSetTask
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, save_checkpoint, make_optimizer, make_scheduler, fix_random_seed, ModelEma)
from libs.utils.train_utils import valid_one_epoch_loss
# from libs.utils.model_utils import count_parameters
from libs.utils import fix_random_seed, ReferringRecall, valid_one_epoch_nlq_singlegpu, valid_one_epoch_cl_single_gpu, final_validate
from libs.cl_methods import on_task_update, on_task_mas_update

################################################################################
def load_best_checkpoint(model, file_folder, file_name, current_task):
    path_best_model = os.path.join(file_folder, file_name)
    if os.path.exists(path_best_model):
        checkpoint_dict = torch.load(path_best_model)
        task_to_load = checkpoint_dict['current_task']
        if task_to_load == current_task:
            model.load_state_dict(checkpoint_dict['state_dict'])
            model.reg_params = checkpoint_dict['reg_params']
    return model

def main(args):
    """main function that handles training / inference"""
    
    global list_val_recall_ii
    list_val_recall_ii = {'val': [], 'test': []}

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
    print(cfg['opt']["learning_rate"])
    # cfg['loader']['num_workers'] *= torch.cuda.device_count()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """2. create dataset / dataloader"""
    path_data = cfg['cl_cfg']['pkl_file']
    with open(path_data, 'rb') as handle:
        data = pickle.load(handle)
    memory_size = cfg['cl_cfg']['memory_size']
    random_order = cfg['cl_cfg']['random_order']
    path_memory = cfg['cl_cfg']['path_memory']

    train_qilDatasetList = QILSetTask(cfg, data['train'], memory_size, shuffle=True, train_enable = True, shuffle_task_order=args.random_order_cl_tasks)
    val_qilDatasetList = QILSetTask(cfg, data['val'], memory_size, shuffle=False, train_enable = False, shuffle_task_order=False)
    current_task = 0
    
    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])

    # if int(os.environ["LOCAL_RANK"]) == 0:
    #     print(model)
    #     count_parameters(model)

    # enable model EMA
    # print("Using model EMA ...")
    # model_ema = ModelEma(model)
    model_ema = None

    gpu_id = int(os.environ["LOCAL_RANK"])
    model = model.to(gpu_id)
    # model = DDP(model, device_ids=[gpu_id])

    # if model_ema is not None:
    #     model_ema = model_ema.to(gpu_id)

    # optimizer
    if cfg['opt']["backbone_lr_weight"] == 1:
        optimizer = make_optimizer(model, cfg['opt'])
    else:
        optimizer = make_optimizer(model, cfg['opt'], head_backbone_group=True)
    # optimizer = make_optimizer(model, cfg['opt'],head_backbone_group=True)
    # schedule
    iter_trainDataloader = iter(train_qilDatasetList)
    num_tasks = train_qilDatasetList.num_tasks
    data, train_loader_i, num_next_queries = next(iter_trainDataloader)
    num_iters_per_epoch = len(train_loader_i)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)
    # metric
    det_eval = ReferringRecall(dataset=cfg["dataset_name"],gt_file=cfg["dataset"]["json_file"])

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(gpu_id))
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if not "head" in k}
            model.load_state_dict(pretrained_dict, strict=False)
            print("initialize head parameters from scratch")
            if args.resume_from_pretrain:
                args.start_epoch = 0
            else:
                args.start_epoch = checkpoint['epoch'] + 1
                try:
                    model.load_state_dict(checkpoint['state_dict'])
                    # model_ema.load_state_dict(checkpoint['state_dict_ema'])
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

    # score_writer = open(os.path.join(ckpt_folder, "eval_results.txt"), mode="w", encoding="utf-8")
    memory_size = cfg['cl_cfg']['memory_size']
    
    for j in range(current_task, num_tasks):
        if j != 0:
            data, train_loader_i, num_next_queries = next(iter_trainDataloader)
        with torch.no_grad():
            best_R1 = valid_one_epoch_cl_single_gpu(val_qilDatasetList, model, 0, j, evaluator=det_eval, tb_writer=tb_writer, print_freq=args.print_freq)
            print('Best init R@1: {} Task: {}'.format(best_R1, j+1))
        
        prev_out_cls_logits_dict = {}
        if model.type_sampling == 'icarl':
            n_classes = model.cls_head.cls_head.conv.out_channels
            for video_list in train_loader_i:
                out_cls_logits, out_offsets, fpn_masks = model(video_list, task_id=j, get_emb=True)
                len_data = len(video_list)
                len_f = len(out_cls_logits)
                for i in range(len_data):
                    video_id = video_list[i]['video_id']
                    prev_out_cls_logits_dict[video_id] = [np.array(torch.sigmoid(out_cls_logits[k][i]).cpu().detach().numpy()) for k in range(len_f)]
            torch.cuda.empty_cache()
        
        for epoch in range(args.start_epoch, max_epochs):
            # train for one epoch
            train_loader_i.sampler.set_epoch(epoch)
            if model.use_adapter:
                model.pre_train_epoch(task_id=j, current_epoch=epoch)
            train_one_epoch(
                train_loader_i,
                model,
                optimizer,
                scheduler,
                epoch,
                model_ema=model_ema,
                clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
                tb_writer=tb_writer,
                print_freq=args.print_freq,
                cl_name=cfg['cl_cfg']['name'], reg_lambda=cfg['cl_cfg']['reg_lambda'],
                prev_out_cls_logits_dict=prev_out_cls_logits_dict,
                current_task_id=j
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
                # losses_tracker = valid_one_epoch_loss(
                #     val_qilDatasetList,
                #     model,
                #     epoch,
                #     tb_writer=tb_writer,
                #     print_freq=args.print_freq / 2
                # )
                # recall_results = valid_one_epoch_nlq_singlegpu(
                #     val_qilDatasetList,
                #     model,
                #     epoch,
                #     evaluator=det_eval,
                #     tb_writer=tb_writer,
                #     print_freq=args.print_freq / 2
                # )
                with torch.no_grad():
                    R1 = valid_one_epoch_cl_single_gpu(
                        val_qilDatasetList, 
                        model, 
                        epoch, 
                        j, 
                        evaluator=det_eval,
                        tb_writer=tb_writer,
                        print_freq=args.print_freq
                        )
                    is_best = R1 >= best_R1
                    best_R1 = max(R1, best_R1)
                    output_best = 'Best R@1: %.3f\n' % (best_R1)
                    print("Best R@1:", output_best)
                    end = time.time()
                    print("All done! Total time: {:0.2f} sec".format(end - start))
                # print("losses_tracker: ", losses_tracker)
                # score_str = ""

                # for key, value in losses_tracker.items():
                #     score_str += '\t{:s} {:.2f} ({:.2f})'.format(
                #         key, value.val, value.avg
                #     )

                # score_writer.write(score_str)
                # score_writer.flush()

            if int(os.environ["LOCAL_RANK"]) == 0:
                save_states = {'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            # 'state_dict_ema': model_ema.module.state_dict(),
                            'current_task': j,
                            'reg_params': model.reg_params,
                            }

                # save_checkpoint(
                #     save_states,
                #     False,
                #     file_folder=ckpt_folder,
                #     file_name='epoch_{:03d}_task_{:02d}.pth.tar'.format(epoch, j)
                # )
                if is_best:
                    save_checkpoint(
                        save_states,
                        False,
                        file_folder=ckpt_folder,
                        file_name='Best_task_{:02d}.pth.tar'.format(j)
                    )
                    print("Save Best Networks for task: {}, epoch: {}".format(save_states['current_task'], save_states['epoch']), flush=True)
                    
        # Compute the number of instances per class that can be saved into the memory after learning the current task. (Mem size/num learned classes).
        if memory_size != 'ALL':
            if torch.cuda.device_count() > 1:
                m = memory_size // 13
            else:
                m = memory_size // 13
        else:
            m = 'ALL'
        
        # Add the new instances to the memory and fit previous instances per class to the new size.
        if memory_size != 0:
            model.add_samples_to_mem(val_qilDatasetList, data, m)
        # Asign the memory to the set of CIL tasks (CILSetTask).
        train_qilDatasetList.memory = model.memory
        # Asign the num of learned classes.
        model.n_known = j + 1
        print('n_known_templates: ',model.n_known)
        # Save the memory
        with open(os.path.join(ckpt_folder, path_memory), 'wb') as handle:
            pickle.dump(model.memory, handle)
        
        model = load_best_checkpoint(model, file_folder=ckpt_folder, file_name='Best_task_{:02d}.pth.tar'.format(j), current_task=j)
        
        with torch.no_grad():
            total_recall_val = final_validate(val_qilDatasetList, model, epoch, j, det_eval, tb_writer=tb_writer, print_freq=args.print_freq, list_val_recall_ii=list_val_recall_ii, type_val='val')
        print('Val total Recall: {}'.format(total_recall_val))
        
        if num_next_queries is not None:
            print('....Update model....')

            # Load the best model achieved for the current task
            model = load_best_checkpoint(model, file_folder=ckpt_folder, file_name='Best_task_{:02d}.pth.tar'.format(j), current_task=j)

            # Calculate the importance of weights for current task
            # EWC Method
            if cfg['cl_cfg']['name'] == 'ewc':
                model.reg_params = on_task_update(train_loader_i, device, optimizer, model)
            elif cfg['cl_cfg']['name'] == 'mas':
                model.reg_params = on_task_mas_update(train_loader_i, device, optimizer, model)
            
            # optimizer
            if cfg['opt']["backbone_lr_weight"] == 1:
                optimizer = make_optimizer(model, cfg['opt'])
            else:
                optimizer = make_optimizer(model, cfg['opt'], head_backbone_group=True)
            scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)
            
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
    parser.add_argument('--random_order_cl_tasks', default=False, type=bool)
    
    args = parser.parse_args()
    main(args)
