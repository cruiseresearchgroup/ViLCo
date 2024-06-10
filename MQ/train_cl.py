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
from libs.datasets import make_dataset, make_data_loader, QILSetTask
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma, validate_loss, valid_one_epoch,
                        valid_one_epoch_cl_single_gpu, final_validate)
from libs.cl_methods import on_task_update, on_task_mas_update
import logging
from eval import valid_performance
from torch.distributed import init_process_group, destroy_process_group
import pickle
import torch.nn.functional as F
################################################################################

def load_best_checkpoint(model, file_folder, file_name, current_task, gpu_id):
    path_best_model = os.path.join(file_folder, file_name)
    if os.path.exists(path_best_model):
        checkpoint_dict = torch.load(path_best_model, map_location=lambda storage, loc: storage.cuda(gpu_id))
        # task_to_load = checkpoint_dict['current_task']
        # if task_to_load == current_task:
        model.load_state_dict(checkpoint_dict['state_dict'])
        model.reg_params = checkpoint_dict['reg_params']
        model = model.cuda()
    return model


def main(args):
    """main function that handles training / inference"""
    
    global list_val_recall_ii
    global list_val_mAP_ii
    list_val_recall_ii = {'val': []}
    list_val_mAP_ii = {'val': []}

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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    """2. create dataset / dataloader"""
    path_data = cfg['cl_cfg']['pkl_file']
    with open(path_data, 'rb') as handle:
        data = pickle.load(handle)
    memory_size = cfg['cl_cfg']['memory_size']
    random_order = cfg['cl_cfg']['random_order']
    path_memory = cfg['cl_cfg']['path_memory']

    train_qilDatasetList = QILSetTask(cfg, data['train'], memory_size, shuffle=True, train_enable = True, shuffle_task_order=random_order)
    val_qilDatasetList = QILSetTask(cfg, data['val'], memory_size, shuffle=False, train_enable = False, shuffle_task_order=False)
    current_task = 0
    
    
    # train_dataset = make_dataset(
    #     cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    # )
    # # update cfg based on dataset attributes (fix to epic-kitchens)
    # train_db_vars = train_dataset.get_attributes()
    # cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # # data loaders
    # train_loader = make_data_loader(
    #     train_dataset, True, rng_generator, **cfg['loader'])
    # val_dataset = make_dataset(
    #     cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    # )
    # # set bs = 1, and disable shuffle
    # val_loader = make_data_loader(
    #     val_dataset, False, None, 1, cfg['loader']['num_workers']
    # )


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
    # num_iters_per_epoch = len(train_loader)
    # scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)
    iter_trainDataloader = iter(train_qilDatasetList)
    num_tasks = train_qilDatasetList.num_tasks
    data, train_loader_i, num_next_classes = next(iter_trainDataloader)
    num_iters_per_epoch = len(train_loader_i)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    # print("Using model EMA ...")
    # model_ema = ModelEma(model)
    # if model_ema is not None:
    #     model_ema = model_ema.to(gpu_id)
    model_ema = None

    # val_db_vars = val_dataset.get_attributes()
    tiou_thresholds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    json_file = './data/ego4d/ego4d_mq_query_incremental_22_all.pkl'
    split = 'val'
    evaluator = ANETdetection(
        json_file,
        split,
        tiou_thresholds = tiou_thresholds,
        use_cl=True
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
            # model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
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
    
    memory_size = cfg['cl_cfg']['memory_size']
    for j in range(current_task, num_tasks):
        if j != 0:
            data, train_loader_i, num_next_queries = next(iter_trainDataloader)
        with torch.no_grad():
            total_R1_0_3, total_R5_0_3, total_R1_0_5, total_R5_0_5, total_mAP = valid_one_epoch_cl_single_gpu(val_qilDatasetList, model, 0, j, evaluator=evaluator, tb_writer=None, logger=logger, print_freq=100)
            if cfg['dataset_name'] == "ego4d" or cfg['dataset_name'] == "ego4d_cl":
                tious = [0.3, 0.5]
                recalls = [1, 5]
                epoch = 0
                best_epoch_of_avgmap = epoch
                best_task_of_avgmap = j
                best_avgmap = total_mAP 
                logger.info(f'Current init Recall 1@0.5 is : [task {best_task_of_avgmap}], [epoch {best_epoch_of_avgmap}], {total_R1_0_5 * 100: .2f} %')
            logger.info(f'Current init Average Map is  : [task {best_task_of_avgmap}], [epoch {best_epoch_of_avgmap}], {best_avgmap * 100: .2f} %')
        
        best_epoch_of_avgmap = -1
        best_avgmap = -10000.0
        best_recall = None
        
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
            if model.use_adapt:
                model.pre_train_epoch(task_id=j, current_epoch=epoch)
            train_one_epoch(
                train_loader_i,
                model,
                optimizer,
                scheduler,
                epoch,
                n_gpu,
                model_ema = model_ema,
                clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
                tb_writer=tb_writer,
                print_freq=args.print_freq, logger=logger,
                cl_name=cfg['cl_cfg']['name'], reg_lambda=cfg['cl_cfg']['reg_lambda'],
                prev_out_cls_logits_dict=prev_out_cls_logits_dict,
                current_task_id=j
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
                    # cur_model = copy.deepcopy(model)
                    # cur_model.load_state_dict(model_ema.module.state_dict())
                    total_R1_0_3, total_R5_0_3, total_R1_0_5, total_R5_0_5, total_mAP = valid_one_epoch_cl_single_gpu(val_qilDatasetList, model, epoch, j,
                                                            evaluator=evaluator, tb_writer=None,
                                                            logger=logger, dataset_name=cfg['dataset_name'], print_freq=100)

                if total_mAP > best_avgmap:
                    best_avgmap = total_mAP
                    best_epoch_of_avgmap = epoch
                    best_task_of_avgmap = j
                    best_recall = [total_R1_0_3, total_R5_0_3, total_R1_0_5, total_R5_0_5]
                    save_states = {
                        'task': j,
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'reg_params': model.reg_params,
                    }
                    # if model_ema is not None:
                    #     save_states['state_dict_ema'] = model_ema.module.state_dict()
                    # save_states['state_dict_ema'] = model.module.state_dict()
                    save_checkpoint(
                        save_states,
                        file_folder=ckpt_folder,
                        file_name='best_task_{:03d}_performance.pth.tar'.format(j)
                    )
                if cfg['dataset_name'] == "ego4d" or cfg['dataset_name'] == "ego4d_cl":
                    tious = [0.3, 0.5]
                    recalls = [1, 5]
                    recall1x5 = best_recall[2]    
                    logger.info(f'Current Best Recall 1@0.5 is : [task {best_task_of_avgmap}], [epoch {best_epoch_of_avgmap}], {recall1x5 * 100: .2f} %')
                logger.info(f'Current Best Average Map is  : [task {best_task_of_avgmap}], [epoch {best_epoch_of_avgmap}], {best_avgmap * 100: .2f} %')
            else:
                if int(os.environ["LOCAL_RANK"]) == 0 and (epoch > max_epochs - 5):
                # if epoch == 11:
                    save_states = {
                        'task': j,
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'reg_params': model.reg_params,
                    }
                    # if model_ema is not None:
                    #     save_states['state_dict_ema'] = model_ema.module.state_dict()
                    # save_states['state_dict_ema'] = model.state_dict()
                    # save_checkpoint(
                    #     save_states,
                    #     file_folder=ckpt_folder,
                    #     file_name='task_{:03d}_epoch_{:03d}.pth.tar'.format(j, epoch)
                    # )
            # ============= infer each epoch ==========
        # Compute the number of instances per class that can be saved into the memory after learning the current task. (Mem size/num learned classes).
        if memory_size != 'ALL':
            if torch.cuda.device_count() > 1:
                m = memory_size // model.cls_head.cls_head.module.conv.out_channels
            else:
                m = memory_size // model.cls_head.cls_head.conv.out_channels
        else:
            m = 'ALL'
        
        # Add the new instances to the memory and fit previous instances per class to the new size.
        if memory_size != 0:
            model.add_samples_to_mem(val_qilDatasetList, data, m)
        # Asign the memory to the set of CIL tasks (CILSetTask).
        train_qilDatasetList.memory = model.memory
        # Asign the num of learned classes.
        model.n_known = len(model.memory)
        print('n_known_classes: ',model.n_known)
        # Save the memory
        with open(os.path.join(ckpt_folder, path_memory), 'wb') as handle:
            pickle.dump(model.memory, handle)
        
        model = load_best_checkpoint(model, file_folder=ckpt_folder, file_name='best_task_{:03d}_performance.pth.tar'.format(j), current_task=j, gpu_id=gpu_id)
        with torch.no_grad():
            total_R1_0_3, total_R5_0_3, total_R1_0_5, total_R5_0_5, total_mAP, BWF_R1_0_5, BWF_mAP = final_validate(val_qilDatasetList, model, epoch, j, evaluator=evaluator, tb_writer=None, logger=logger, print_freq=100, list_val_recall_ii=list_val_recall_ii, list_val_mAP_ii=list_val_mAP_ii, type_val='val')
        if cfg['dataset_name'] == "ego4d" or cfg['dataset_name'] == "ego4d_cl":
            tious = [0.1, 0.2, 0.3, 0.4, 0.5]
            recalls = [1, 5]
            recall1x5 = total_R1_0_5   
            logger.info(f'Final Average Recall 1@0.5 is : [task {best_task_of_avgmap}], {recall1x5 * 100: .2f} %')
        logger.info(f'Final Average Map is  : [task {best_task_of_avgmap}], {total_mAP * 100: .2f} %')
        
        if num_next_classes is not None:
            print('....Update model....')

            # Load the best model achieved for the current task
            # model = load_best_checkpoint(model, file_folder=ckpt_folder, file_name='best_task_{:03d}_performance.pth.tar'.format(j), current_task=j, gpu_id=gpu_id)
            model.augment_classification(num_next_classes, device)
            
            # Calculate the importance of weights for current task
            # EWC Method
            if cfg['cl_cfg']['name'] == 'ewc':
                model.reg_params = on_task_update(train_loader_i, device, optimizer, model)
            elif cfg['cl_cfg']['name'] == 'mas':
                model.reg_params = on_task_mas_update(train_loader_i, device, optimizer, model)
            
            # optimizer
            optimizer = make_optimizer(model, cfg['opt'])
            scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)
        
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
