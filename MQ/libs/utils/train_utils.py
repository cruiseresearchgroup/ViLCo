import os
import shutil
import time
import pickle

import numpy as np
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from .get_retrieval_performance import evaluation_retrieval
from .get_detect_performance import evaluation_detection

from .lr_schedulers import LinearWarmupMultiStepLR, LinearWarmupCosineAnnealingLR
from .postprocessing import postprocess_results
from ..modeling import MaskedConv1D, Scale, AffineDropPath, LayerNorm
from ..modeling.modeling_xlnet_x import XLNetModel, XLNetLMHeadModel
from functools import partial
import logging
import time
from .metrics import ANETdetection
from torch.nn.parallel import DistributedDataParallel as DDP
from ..cl_methods.EWC import get_regularized_loss
from ..cl_methods.MAS import get_mas_regularized_loss
from torch.autograd import Variable
from ..modeling.meta_archs import BiasLayer


################################################################################
def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator


def save_checkpoint(state, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif 'xlnet' in pn and 'norm' not in pn:
                decay.add(fpn)
            elif 'xlnet' in pn and 'norm' in pn:
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    remain_params = param_dict.keys() - union_params
    if model.use_adapt:
        for mn, m in model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if 'pets' in fpn:
                    param_dict[fpn] = p
                        
    # assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    # assert len(param_dict.keys() - union_params) == 0, \
    #     "parameters %s were not separated into either decay/no_decay set!" \
    #     % (str(param_dict.keys() - union_params), )
    # for pn_1 in sorted(list(decay)):
    #     for pn_2 in sorted(list(remain_params)):
    #         if id(param_dict[pn_1]) == id(param_dict[pn_2]):
    #             param_dict.pop(pn_1)
    
    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        {"params": [param_dict[pn] for pn in sorted(list(remain_params))], "weight_decay": optimizer_config['weight_decay']}
    ]
    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # Multi step
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = LinearWarmupMultiStepLR(
                optimizer,
                warmup_steps,
                steps,
                gamma=optimizer_config["schedule_gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )

        elif optimizer_config["schedule_type"] == "multistep":
            # step every some epochs
            steps = [num_iters_per_epoch * step for step in optimizer_config["schedule_steps"]]
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                steps,
                gamma=schedule_config["gamma"],
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn, update_fn_1=None):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                # print(ema_v.size() != model_v.size(), ema_v.size(),  model_v.size())
                if ema_v.size() != model_v.size():
                    ema_v = model_v
                    ema_v.copy_(ema_v)
                else:
                    ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m) 

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


################################################################################
def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    n_gpu,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 20,
    logger=None,
    cl_name=None, 
    reg_lambda=0.0,
    prev_out_cls_logits_dict=None,
    current_task_id=0
):
    """Training the model for one epoch"""
    # set up meters
    gpu_id = int(os.environ["LOCAL_RANK"])
    # model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()
    model.compute_means = True if model.cl_name == 'icarl' else False

    # ======== test two stage ==========
    # for n, p in model.named_parameters():
    #     if 'cls_head' not in n:
    #         p.requires_grad = False
    # ======== test two stage ==========


    # main training loop
    if int(os.environ["LOCAL_RANK"]) == 0:
        print("\n[Train]: [GPU{:d}] Epoch {:d} started".format(gpu_id, curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # video list 
        # import ipdb;ipdb.set_trace()
        # zero out optim
        optimizer.zero_grad(set_to_none=True)

        # forward / backward the model
        batch_size = len(video_list)
        prev_out_cls_logits = []
        for i in range(batch_size):
            video_id = video_list[i]['video_id']
            if prev_out_cls_logits_dict is not None and (video_id in prev_out_cls_logits_dict.keys()):
                prev_out_cls_logits.append(prev_out_cls_logits_dict[video_id])
        # prev_out_cls_logits = prev_out_cls_logits_dict[video_id] 
        losses = model(video_list, task_id=current_task_id, prev_out_cls_logits=prev_out_cls_logits)
        if n_gpu > 1:
            losses['final_loss'] = losses['final_loss'].mean() # mean() to average on multi-gpu
        if cl_name == 'ewc':
            loss = get_regularized_loss(losses['final_loss'], model, reg_lambda)
            # loss = Variable(loss, requires_grad=True)
            loss.backward()
        elif cl_name == 'mas':
            loss = get_mas_regularized_loss(losses['final_loss'], model, reg_lambda)
            loss.backward()
        else:
            losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()
        # if model_ema is not None:
        #     model_ema.update(model)
        
        if model.use_adapt:
            model.post_train_step()

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/GPU_{}/learning_rate'.format(gpu_id),
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/GPU_{}/all_losses'.format(gpu_id),
                    tag_dict,
                    global_step
                )
                # final loss
                tb_writer.add_scalar(
                    'train/GPU_{}/final_loss'.format(gpu_id),
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # print to terminal
            block1 = '[GPU{:d}], Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                gpu_id, curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4  += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )
            logger.info('\t'.join([block1, block2, block3, block4]))

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    logger.info("[GPU{:d}] [Train]: Epoch {:d} finished with lr={:.8f}\n".format(gpu_id, curr_epoch, lr))
    return

def train_bic_one_epoch(
    train_train_loader,
    train_val_loader,
    len_train_train_data,
    len_train_val_data,
    stage,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    n_gpu,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 20,
    logger=None,
    cl_name=None, 
    reg_lambda=0.0,
    prev_out_cls_logits_dict=None,
    current_task_id=0
):
    """Training the model for one epoch"""
    # set up meters
    gpu_id = int(os.environ["LOCAL_RANK"])
    # model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
    batch_time = AverageMeter()
    losses_tracker = {}
    # switch to train mode
    model.train()
    model.compute_means = False

    # ======== test two stage ==========
    # for n, p in model.named_parameters():
    #     if 'cls_head' not in n:
    #         p.requires_grad = False
    # ======== test two stage ==========

    if stage == 0:
        len_data = len_train_train_data
        dataloader_cil = train_train_loader
        for bias_layer in model.list_bias_layers:
            bias_layer.eval()
    else:
        len_data = len_train_val_data
        dataloader_cil = train_val_loader
        for bias_layer in model.list_bias_layers:
            bias_layer.train()
            
    # number of iterations per epoch
    num_iters = len(dataloader_cil)

    # main training loop
    if int(os.environ["LOCAL_RANK"]) == 0:
        print("\n[Train]: [GPU{:d}] Epoch {:d} started".format(gpu_id, curr_epoch))
    start = time.time()
    
    for iter_idx, video_list in enumerate(dataloader_cil, 0):
        # video list 
        # import ipdb;ipdb.set_trace()
        # zero out optim
        optimizer.zero_grad(set_to_none=True)

        # forward / backward the model
        batch_size = len(video_list)
        prev_out_cls_logits = []
        for i in range(batch_size):
            video_id = video_list[i]['video_id']
            prev_out_cls_logits.append(prev_out_cls_logits_dict[video_id])
        prev_out_cls_logits = prev_out_cls_logits_dict[video_id] 
        losses = model(video_list, task_id=current_task_id, prev_out_cls_logits=prev_out_cls_logits)
        if n_gpu > 1:
            losses['final_loss'] = losses['final_loss'].mean() # mean() to average on multi-gpu
        if cl_name == 'ewc':
            loss = get_regularized_loss(losses['final_loss'], model, reg_lambda)
            # loss = Variable(loss, requires_grad=True)
            loss.backward()
        elif cl_name == 'mas':
            loss = get_mas_regularized_loss(losses['final_loss'], model, reg_lambda)
            loss.backward()
        else:
            losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()
        # if model_ema is not None:
        #     model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/GPU_{}/learning_rate'.format(gpu_id),
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/GPU_{}/all_losses'.format(gpu_id),
                    tag_dict,
                    global_step
                )
                # final loss
                tb_writer.add_scalar(
                    'train/GPU_{}/final_loss'.format(gpu_id),
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # print to terminal
            block1 = '[GPU{:d}], Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                gpu_id, curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4  += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )
            logger.info('\t'.join([block1, block2, block3, block4]))

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    logger.info("[GPU{:d}] [Train]: Epoch {:d} finished with lr={:.8f}\n".format(gpu_id, curr_epoch, lr))
    return

def validate_loss(
    val_loader,
    model,
    curr_epoch,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 20,
    logger=None
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(val_loader)
    # switch to train mode
    model.eval()
    val_loss = 0.0
    num_samples = 0
    # main training loop
    print("\n[Eval]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    with torch.no_grad():
        for iter_idx, video_list in enumerate(val_loader, 0):
            
            # forward / backward the model
            losses = model(video_list)

            # printing (only check the stats when necessary to avoid extra cost)
            if (iter_idx != 0) and (iter_idx % print_freq) == 0:
                # measure elapsed time (sync all kernels)
                torch.cuda.synchronize()
                batch_time.update((time.time() - start) / print_freq)
                start = time.time()

                # track all losses
                for key, value in losses.items():
                    # init meter if necessary
                    if key not in losses_tracker:
                        losses_tracker[key] = AverageMeter()
                    # update
                    losses_tracker[key].update(value.item())

                # log to tensor board
                global_step = curr_epoch * num_iters + iter_idx
                if tb_writer is not None:
                    # all losses
                    tag_dict = {}
                    for key, value in losses_tracker.items():
                        if key != "final_loss":
                            tag_dict[key] = value.val
                    tb_writer.add_scalars(
                        'eval/all_losses',
                        tag_dict,
                        global_step
                    )
                    # final loss
                    tb_writer.add_scalar(
                        'eval/final_loss',
                        losses_tracker['final_loss'].val,
                        global_step
                    )
            
            val_loss += len(video_list) * losses['final_loss'].item() 
            num_samples += len(video_list)

        val_loss /= num_samples
        # finish up and print
        logger.info("[Eval]: Epoch {:d} finished with val loss {:.5f} \n".format(curr_epoch, val_loss))
        return val_loss




def valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
    logger=None,
    dataset_name=None
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    torch.set_grad_enabled(False)
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():

            output = model(video_list, is_training=False)
            # output, video_list, points, fpn_masks, out_cls_logits, out_offsets = model(video_list, hidden_state=True)
            # upack the results into ANet format
            # output = output1.detach().clone()
            # del output1
            # torch.cuda.empty_cache()
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])



            # evaluator_cur = ANETdetection(
            #     'data/ego4d/ego4d_clip_annotations.json',
            #     'val',
            #     tiou_thresholds = np.linspace(0.1, 0.5, 5),
            #     debug_video_id=results['video-id']  
            # )
            # results_cur = {}   
            # results_cur['video-id'] = results['video-id'][-200:]     
            # results_cur['t-start'] = output[vid_idx]['segments'][:, 0].numpy()
            # results_cur['t-end'] = output[vid_idx]['segments'][:, 1].numpy()
            # results_cur['label'] = output[vid_idx]['labels'].numpy()
            # results_cur['score'] = output[vid_idx]['scores'].numpy()
            # mAP, avg_mAP, tiou_thresholds = evaluator_cur.evaluate(results_cur, verbose=False)
            # # for tiou, tiou_mAP in zip(tiou_thresholds, mAP):
            #         # logger.info(f'tIoU = {tiou:.1f}: mAP = {tiou_mAP * 100:.2f} %')
            # # logger.info(f'Average Map is :{avg_mAP * 100: .2f} %')
            # if avg_mAP * 100 < 25.0:
            #     print(f'Average Map is :{avg_mAP * 100: .2f} %')
            #     import ipdb;ipdb.set_trace()


            # printing
            if (iter_idx != 0) and iter_idx % (print_freq) == 0:
                # measure elapsed time (sync all kernels)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                batch_time.update((time.time() - start) / print_freq)
                start = time.time()

                # print timing
                logger.info('Test: [{0:05d}/{1:05d}]\t'
                    'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                    iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()
    
    eval_result = None
    if dataset_name == "ego4d":
        #retrieval ######################### SAVE######################### SAVE######################### SAVE######################### SAVE######################### SAVE 
        classes = {'use_phone': 16, 'water_soil_/_plants_/_crops': 59, 'clean_/_wipe_a_table_or_kitchen_counter': 29, 'walk_down_stairs_/_walk_up_stairs': 30, 'arrange_/_organize_other_items': 24, 'clean_/_wipe_other_surface_or_object': 6, 'fill_a_pot_/_bottle_/_container_with_water': 64, 'use_a_laptop_/_computer': 37, 'knead_/_shape_/_roll-out_dough': 22, 'cut_dough': 25, 'fry_dough': 57, 'converse_/_interact_with_someone': 11, 'stir_/_mix_food_while_cooking': 21, 'wash_dishes_/_utensils_/_bakeware_etc.': 68, 'turn-on_/_light_the_stove_burner': 9, 'serve_food_onto_a_plate': 67, 'chop_/_cut_wood_pieces_using_tool': 82, 'cut_/_trim_grass_with_other_tools': 92, 'trim_hedges_or_branches': 100, 'browse_through_groceries_or_food_items_on_rack_/_shelf': 32, 'read_a_book_/_magazine_/_shopping_list_etc.': 28, 'take_photo_/_record_video_with_a_camera': 0, 'pay_at_billing_counter': 42, 'stand_in_the_queue_/_line_at_a_shop_/_supermarket': 53, 'browse_through_other_items_on_rack_/_shelf': 50, 'browse_through_clothing_items_on_rack_/_shelf_/_hanger': 2, 'look_at_clothes_in_the_mirror': 83, '"try-out_/_wear_accessories_(e.g._tie,_belt,_scarf)"': 102, 'put_away_(or_take_out)_dishes_/_utensils_in_storage': 81, 'clean_/_wipe_kitchen_appliance': 23, 'wash_vegetable_/_fruit_/_food_item': 95, '"cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat"': 75, 'cut_other_item_using_tool': 27, 'drill_into_wall_/_wood_/_floor_/_metal': 19, 'use_hammer_/_nail-gun_to_fix_nail': 34, 'weigh_food_/_ingredient_using_a_weighing_scale': 54, 'pack_food_items_/_groceries_into_bags_/_boxes': 41, 'drink_beverage': 65, 'withdraw_money_from_atm_/_operate_atm': 3, 'put_away_(or_take_out)_food_items_in_the_fridge': 39, 'interact_or_play_with_pet_/_animal': 101, 'put_away_(or_take_out)_ingredients_in_storage': 7, '"try-out_/_wear_clothing_items_(e.g._shirt,_jeans,_sweater)"': 77, 'throw_away_trash_/_put_trash_in_trash_can': 8, 'tie_up_branches_/_plants_with_string': 103, 'remove_weeds_from_ground': 85, 'collect_/_rake_dry_leaves_on_ground': 91, 'harvest_vegetables_/_fruits_/_crops_from_plants_on_the_ground': 86, 'place_items_in_shopping_cart': 31, 'write_notes_in_a_paper_/_book': 108, 'wash_hands': 5, 'pack_other_items_into_bags_/_boxes': 73, 'pack_soil_into_the_ground_or_a_pot_/_container': 47, 'plant_seeds_/_plants_/_flowers_into_ground': 48, '"level_ground_/_soil_(eg._using_rake,_shovel,_etc)"': 46, 'dig_or_till_the_soil_with_a_hoe_or_other_tool': 45, 'cut_tree_branch': 90, 'measure_wooden_item_using_tape_/_ruler': 35, 'mark_item_with_pencil_/_pen_/_marker': 36, 'compare_two_clothing_items': 97, 'do_some_exercise': 80, 'watch_television': 17, 'taste_food_while_cooking': 96, 'rinse_/_drain_other_food_item_in_sieve_/_colander': 71, 'use_a_vacuum_cleaner_to_clean': 15, 'fix_other_item': 20, 'smooth_wood_using_sandpaper_/_sander_/_tool': 88, 'dig_or_till_the_soil_by_hand': 98, 'hang_clothes_in_closet_/_on_hangers': 1, 'clean_/_wipe_/_oil_metallic_item': 72, 'fix_bonnet_/_engine_of_car': 107, 'hang_clothes_to_dry': 109, 'cut_/_trim_grass_with_a_lawnmower': 76, 'fold_clothes_/_sheets': 56, 'dismantle_other_item': 18, 'fix_/_remove_/_replace_a_tire_or_wheel': 84, 'move_/_shift_/_arrange_small_tools': 78, 'make_coffee_or_tea_/_use_a_coffee_machine': 63, 'play_board_game_or_card_game': 60, 'count_money_before_paying': 40, 'enter_a_supermarket_/_shop': 49, 'exit_a_supermarket_/_shop': 51, 'play_a_video_game': 79, 'arrange_pillows_on_couch_/_chair': 104, '"make_the_bed_/_arrange_pillows,_sheets_etc._on_bed"': 105, 'clean_/_sweep_floor_with_broom': 61, 'arrange_/_organize_clothes_in_closet/dresser': 55, 'load_/_unload_a_washing_machine_or_dryer': 89, 'move_/_shift_around_construction_material': 70, '"put_on_safety_equipment_(e.g._gloves,_helmet,_safety_goggles)"': 52, 'cut_open_a_package_(e.g._with_scissors)': 66, 'stir_/_mix_ingredients_in_a_bowl_or_pan_(before_cooking)': 4, 'fry_other_food_item': 38, 'eat_a_snack': 62, 'drive_a_vehicle': 99, 'arrange_/_organize_items_in_fridge': 10, 'browse_through_accessories_on_rack_/_shelf': 43, 'fix_wiring': 26, 'prepare_or_apply_cement_/_concrete_/_mortar': 69, 'put_food_into_the_oven_to_bake': 106, 'peel_a_fruit_or_vegetable': 74, 'smoke_cigar_/_cigarette_/_vape': 93, 'paint_using_paint_brush_/_roller': 14, 'climb_up_/_down_a_ladder': 12, 'cut_thread_/_paper_/_cardboard_using_scissors_/_knife_/_cutter': 44, 'plaster_wall_/_surface': 13, 'fix_pipe_/_plumbing': 87, '"clean_/_repair_small_equipment_(mower,_trimmer_etc.)"': 33, 'remove_food_from_the_oven': 58, 'iron_clothes_or_sheets': 94}
        idx_classes = {}
        for key, value in classes.items():
            idx_classes[value] = key
        from collections import defaultdict
        from tqdm import tqdm
        import json

        # # test upperbound for cls
        actionformer_res = defaultdict(list)
        
        for video_id, t_start, t_end, label, score in tqdm(zip(results['video-id'], results['t-start'], results['t-end'], results['label'], results['score'])):
            # test upperbound for cls
            t_start, t_end, label, score = t_start.item(), t_end.item(), label.item(), score.item()
            actionformer_res[video_id].append({
                'segment': [t_start, t_end], 'score': score, 'label': idx_classes[label]
            })
        save_obj = {"version": "1.0", "external_data": "", 'results': actionformer_res}
        
        if not os.path.exists("retrieval_json"):
            os.makedirs("retrieval_json")
        now = time.time() 
        now = str(int(round(now * 1000)))
        pred_file = os.path.join("retrieval_json", f"results_{output_file}_{now}.json")

        with open(pred_file, 'w') as f:
            json.dump(save_obj, f)
        eval_result = evaluation_retrieval(gt="data/ego4d/ego4d_clip_annotations.json",
                            pred=f"{pred_file}",
                            subset="val",
                            tiou=[0.1, 0.2, 0.3, 0.4, 0.5],
                            )
        tious = [0.1, 0.2, 0.3, 0.4, 0.5]
        recalls = [1, 5]
        for i, t in enumerate(tious):
            for j, r in enumerate(recalls):
                recall = eval_result[i, j]
                logger.info(f'Rank {r}x @ tIoU {t} is {recall}')
        os.remove(pred_file)
        ######################### SAVE######################### SAVE######################### SAVE######################### SAVE######################### SAVE

    assert evaluator is not None
    if (ext_score_file is not None) and isinstance(ext_score_file, str):
        results = postprocess_results(results, ext_score_file)
    # call the evaluator
    mAP, avg_mAP, tiou_thresholds = evaluator.evaluate(results, verbose=False)
    for tiou, tiou_mAP in zip(tiou_thresholds, mAP):
        logger.info(f'tIoU = {tiou:.1f}: mAP = {tiou_mAP * 100:.2f} %')
    logger.info(f'Average Map is :{avg_mAP * 100: .2f} %')

    return mAP, avg_mAP, tiou_thresholds, eval_result



def infer_one_epoch(
    val_loader,
    model,
    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            # import ipdb;ipdb.set_trace()
            output = model(video_list, is_training=False)
            # upack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()
    
    classes = {'use_phone': 16, 'water_soil_/_plants_/_crops': 59, 'clean_/_wipe_a_table_or_kitchen_counter': 29, 'walk_down_stairs_/_walk_up_stairs': 30, 'arrange_/_organize_other_items': 24, 'clean_/_wipe_other_surface_or_object': 6, 'fill_a_pot_/_bottle_/_container_with_water': 64, 'use_a_laptop_/_computer': 37, 'knead_/_shape_/_roll-out_dough': 22, 'cut_dough': 25, 'fry_dough': 57, 'converse_/_interact_with_someone': 11, 'stir_/_mix_food_while_cooking': 21, 'wash_dishes_/_utensils_/_bakeware_etc.': 68, 'turn-on_/_light_the_stove_burner': 9, 'serve_food_onto_a_plate': 67, 'chop_/_cut_wood_pieces_using_tool': 82, 'cut_/_trim_grass_with_other_tools': 92, 'trim_hedges_or_branches': 100, 'browse_through_groceries_or_food_items_on_rack_/_shelf': 32, 'read_a_book_/_magazine_/_shopping_list_etc.': 28, 'take_photo_/_record_video_with_a_camera': 0, 'pay_at_billing_counter': 42, 'stand_in_the_queue_/_line_at_a_shop_/_supermarket': 53, 'browse_through_other_items_on_rack_/_shelf': 50, 'browse_through_clothing_items_on_rack_/_shelf_/_hanger': 2, 'look_at_clothes_in_the_mirror': 83, '"try-out_/_wear_accessories_(e.g._tie,_belt,_scarf)"': 102, 'put_away_(or_take_out)_dishes_/_utensils_in_storage': 81, 'clean_/_wipe_kitchen_appliance': 23, 'wash_vegetable_/_fruit_/_food_item': 95, '"cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat"': 75, 'cut_other_item_using_tool': 27, 'drill_into_wall_/_wood_/_floor_/_metal': 19, 'use_hammer_/_nail-gun_to_fix_nail': 34, 'weigh_food_/_ingredient_using_a_weighing_scale': 54, 'pack_food_items_/_groceries_into_bags_/_boxes': 41, 'drink_beverage': 65, 'withdraw_money_from_atm_/_operate_atm': 3, 'put_away_(or_take_out)_food_items_in_the_fridge': 39, 'interact_or_play_with_pet_/_animal': 101, 'put_away_(or_take_out)_ingredients_in_storage': 7, '"try-out_/_wear_clothing_items_(e.g._shirt,_jeans,_sweater)"': 77, 'throw_away_trash_/_put_trash_in_trash_can': 8, 'tie_up_branches_/_plants_with_string': 103, 'remove_weeds_from_ground': 85, 'collect_/_rake_dry_leaves_on_ground': 91, 'harvest_vegetables_/_fruits_/_crops_from_plants_on_the_ground': 86, 'place_items_in_shopping_cart': 31, 'write_notes_in_a_paper_/_book': 108, 'wash_hands': 5, 'pack_other_items_into_bags_/_boxes': 73, 'pack_soil_into_the_ground_or_a_pot_/_container': 47, 'plant_seeds_/_plants_/_flowers_into_ground': 48, '"level_ground_/_soil_(eg._using_rake,_shovel,_etc)"': 46, 'dig_or_till_the_soil_with_a_hoe_or_other_tool': 45, 'cut_tree_branch': 90, 'measure_wooden_item_using_tape_/_ruler': 35, 'mark_item_with_pencil_/_pen_/_marker': 36, 'compare_two_clothing_items': 97, 'do_some_exercise': 80, 'watch_television': 17, 'taste_food_while_cooking': 96, 'rinse_/_drain_other_food_item_in_sieve_/_colander': 71, 'use_a_vacuum_cleaner_to_clean': 15, 'fix_other_item': 20, 'smooth_wood_using_sandpaper_/_sander_/_tool': 88, 'dig_or_till_the_soil_by_hand': 98, 'hang_clothes_in_closet_/_on_hangers': 1, 'clean_/_wipe_/_oil_metallic_item': 72, 'fix_bonnet_/_engine_of_car': 107, 'hang_clothes_to_dry': 109, 'cut_/_trim_grass_with_a_lawnmower': 76, 'fold_clothes_/_sheets': 56, 'dismantle_other_item': 18, 'fix_/_remove_/_replace_a_tire_or_wheel': 84, 'move_/_shift_/_arrange_small_tools': 78, 'make_coffee_or_tea_/_use_a_coffee_machine': 63, 'play_board_game_or_card_game': 60, 'count_money_before_paying': 40, 'enter_a_supermarket_/_shop': 49, 'exit_a_supermarket_/_shop': 51, 'play_a_video_game': 79, 'arrange_pillows_on_couch_/_chair': 104, '"make_the_bed_/_arrange_pillows,_sheets_etc._on_bed"': 105, 'clean_/_sweep_floor_with_broom': 61, 'arrange_/_organize_clothes_in_closet/dresser': 55, 'load_/_unload_a_washing_machine_or_dryer': 89, 'move_/_shift_around_construction_material': 70, '"put_on_safety_equipment_(e.g._gloves,_helmet,_safety_goggles)"': 52, 'cut_open_a_package_(e.g._with_scissors)': 66, 'stir_/_mix_ingredients_in_a_bowl_or_pan_(before_cooking)': 4, 'fry_other_food_item': 38, 'eat_a_snack': 62, 'drive_a_vehicle': 99, 'arrange_/_organize_items_in_fridge': 10, 'browse_through_accessories_on_rack_/_shelf': 43, 'fix_wiring': 26, 'prepare_or_apply_cement_/_concrete_/_mortar': 69, 'put_food_into_the_oven_to_bake': 106, 'peel_a_fruit_or_vegetable': 74, 'smoke_cigar_/_cigarette_/_vape': 93, 'paint_using_paint_brush_/_roller': 14, 'climb_up_/_down_a_ladder': 12, 'cut_thread_/_paper_/_cardboard_using_scissors_/_knife_/_cutter': 44, 'plaster_wall_/_surface': 13, 'fix_pipe_/_plumbing': 87, '"clean_/_repair_small_equipment_(mower,_trimmer_etc.)"': 33, 'remove_food_from_the_oven': 58, 'iron_clothes_or_sheets': 94}
    idx_classes = {}
    for key, value in classes.items():
        idx_classes[value] = key
    ######################### SAVE######################### SAVE######################### SAVE######################### SAVE######################### SAVE
    from collections import defaultdict
    from tqdm import tqdm
    import json

    actionformer_res = defaultdict(list)
    for video_id, t_start, t_end, label, score in tqdm(zip(results['video-id'], results['t-start'], results['t-end'], results['label'], results['score'])):
        t_start, t_end, label, score = t_start.item(), t_end.item(), label.item(), score.item()
        actionformer_res[video_id].append({
            'segment': [t_start, t_end], 'score': score, 'label': idx_classes[label]
        })
    save_obj = {"version": "1.0", "external_data": "", 'results': actionformer_res}

    with open('submission.json', 'w') as f:
        json.dump(save_obj, f)

    return 0





def infer_one_epoch_ensemble(
    val_loader,
    model1,
    model2,
    model3,
    model4,
    model5,

    curr_epoch,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
    logger=None
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    # switch to evaluate mode
    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()

    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            # import ipdb;ipdb.set_trace()
            video_list, points, fpn_masks, out_cls_logits1, out_offsets1 = model1(video_list, ensemble=True, is_training=False)
            video_list, points, fpn_masks, out_cls_logits2, out_offsets2 = model2(video_list, ensemble=True, is_training=False)
            video_list, points, fpn_masks, out_cls_logits3, out_offsets3 = model3(video_list, ensemble=True, is_training=False)
            video_list, points, fpn_masks, out_cls_logits4, out_offsets4 = model4(video_list, ensemble=True, is_training=False)
            video_list, points, fpn_masks, out_cls_logits5, out_offsets5 = model5(video_list, ensemble=True, is_training=False)
            
            out_cls_logits = []
            out_offsets = []
            for i in range(len(out_cls_logits1)):
                out_cls_logits.append((out_cls_logits1[i] + out_cls_logits2[i] + out_cls_logits3[i] + out_cls_logits4[i] + out_cls_logits5[i]) / 5.0)
                out_offsets.append((out_offsets1[i] + out_offsets2[i] + out_offsets3[i] + out_offsets4[i] + out_offsets5[i]) / 5.0)
            output = model1.module.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets,
                None, None
            )
            # upack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # logger.info timing
            logger.info('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()
    
    classes = {'use_phone': 16, 'water_soil_/_plants_/_crops': 59, 'clean_/_wipe_a_table_or_kitchen_counter': 29, 'walk_down_stairs_/_walk_up_stairs': 30, 'arrange_/_organize_other_items': 24, 'clean_/_wipe_other_surface_or_object': 6, 'fill_a_pot_/_bottle_/_container_with_water': 64, 'use_a_laptop_/_computer': 37, 'knead_/_shape_/_roll-out_dough': 22, 'cut_dough': 25, 'fry_dough': 57, 'converse_/_interact_with_someone': 11, 'stir_/_mix_food_while_cooking': 21, 'wash_dishes_/_utensils_/_bakeware_etc.': 68, 'turn-on_/_light_the_stove_burner': 9, 'serve_food_onto_a_plate': 67, 'chop_/_cut_wood_pieces_using_tool': 82, 'cut_/_trim_grass_with_other_tools': 92, 'trim_hedges_or_branches': 100, 'browse_through_groceries_or_food_items_on_rack_/_shelf': 32, 'read_a_book_/_magazine_/_shopping_list_etc.': 28, 'take_photo_/_record_video_with_a_camera': 0, 'pay_at_billing_counter': 42, 'stand_in_the_queue_/_line_at_a_shop_/_supermarket': 53, 'browse_through_other_items_on_rack_/_shelf': 50, 'browse_through_clothing_items_on_rack_/_shelf_/_hanger': 2, 'look_at_clothes_in_the_mirror': 83, '"try-out_/_wear_accessories_(e.g._tie,_belt,_scarf)"': 102, 'put_away_(or_take_out)_dishes_/_utensils_in_storage': 81, 'clean_/_wipe_kitchen_appliance': 23, 'wash_vegetable_/_fruit_/_food_item': 95, '"cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat"': 75, 'cut_other_item_using_tool': 27, 'drill_into_wall_/_wood_/_floor_/_metal': 19, 'use_hammer_/_nail-gun_to_fix_nail': 34, 'weigh_food_/_ingredient_using_a_weighing_scale': 54, 'pack_food_items_/_groceries_into_bags_/_boxes': 41, 'drink_beverage': 65, 'withdraw_money_from_atm_/_operate_atm': 3, 'put_away_(or_take_out)_food_items_in_the_fridge': 39, 'interact_or_play_with_pet_/_animal': 101, 'put_away_(or_take_out)_ingredients_in_storage': 7, '"try-out_/_wear_clothing_items_(e.g._shirt,_jeans,_sweater)"': 77, 'throw_away_trash_/_put_trash_in_trash_can': 8, 'tie_up_branches_/_plants_with_string': 103, 'remove_weeds_from_ground': 85, 'collect_/_rake_dry_leaves_on_ground': 91, 'harvest_vegetables_/_fruits_/_crops_from_plants_on_the_ground': 86, 'place_items_in_shopping_cart': 31, 'write_notes_in_a_paper_/_book': 108, 'wash_hands': 5, 'pack_other_items_into_bags_/_boxes': 73, 'pack_soil_into_the_ground_or_a_pot_/_container': 47, 'plant_seeds_/_plants_/_flowers_into_ground': 48, '"level_ground_/_soil_(eg._using_rake,_shovel,_etc)"': 46, 'dig_or_till_the_soil_with_a_hoe_or_other_tool': 45, 'cut_tree_branch': 90, 'measure_wooden_item_using_tape_/_ruler': 35, 'mark_item_with_pencil_/_pen_/_marker': 36, 'compare_two_clothing_items': 97, 'do_some_exercise': 80, 'watch_television': 17, 'taste_food_while_cooking': 96, 'rinse_/_drain_other_food_item_in_sieve_/_colander': 71, 'use_a_vacuum_cleaner_to_clean': 15, 'fix_other_item': 20, 'smooth_wood_using_sandpaper_/_sander_/_tool': 88, 'dig_or_till_the_soil_by_hand': 98, 'hang_clothes_in_closet_/_on_hangers': 1, 'clean_/_wipe_/_oil_metallic_item': 72, 'fix_bonnet_/_engine_of_car': 107, 'hang_clothes_to_dry': 109, 'cut_/_trim_grass_with_a_lawnmower': 76, 'fold_clothes_/_sheets': 56, 'dismantle_other_item': 18, 'fix_/_remove_/_replace_a_tire_or_wheel': 84, 'move_/_shift_/_arrange_small_tools': 78, 'make_coffee_or_tea_/_use_a_coffee_machine': 63, 'play_board_game_or_card_game': 60, 'count_money_before_paying': 40, 'enter_a_supermarket_/_shop': 49, 'exit_a_supermarket_/_shop': 51, 'play_a_video_game': 79, 'arrange_pillows_on_couch_/_chair': 104, '"make_the_bed_/_arrange_pillows,_sheets_etc._on_bed"': 105, 'clean_/_sweep_floor_with_broom': 61, 'arrange_/_organize_clothes_in_closet/dresser': 55, 'load_/_unload_a_washing_machine_or_dryer': 89, 'move_/_shift_around_construction_material': 70, '"put_on_safety_equipment_(e.g._gloves,_helmet,_safety_goggles)"': 52, 'cut_open_a_package_(e.g._with_scissors)': 66, 'stir_/_mix_ingredients_in_a_bowl_or_pan_(before_cooking)': 4, 'fry_other_food_item': 38, 'eat_a_snack': 62, 'drive_a_vehicle': 99, 'arrange_/_organize_items_in_fridge': 10, 'browse_through_accessories_on_rack_/_shelf': 43, 'fix_wiring': 26, 'prepare_or_apply_cement_/_concrete_/_mortar': 69, 'put_food_into_the_oven_to_bake': 106, 'peel_a_fruit_or_vegetable': 74, 'smoke_cigar_/_cigarette_/_vape': 93, 'paint_using_paint_brush_/_roller': 14, 'climb_up_/_down_a_ladder': 12, 'cut_thread_/_paper_/_cardboard_using_scissors_/_knife_/_cutter': 44, 'plaster_wall_/_surface': 13, 'fix_pipe_/_plumbing': 87, '"clean_/_repair_small_equipment_(mower,_trimmer_etc.)"': 33, 'remove_food_from_the_oven': 58, 'iron_clothes_or_sheets': 94}
    idx_classes = {}
    for key, value in classes.items():
        idx_classes[value] = key
    ######################### SAVE######################### SAVE######################### SAVE######################### SAVE######################### SAVE
    from collections import defaultdict
    from tqdm import tqdm
    import json

    actionformer_res = defaultdict(list)
    for video_id, t_start, t_end, label, score in tqdm(zip(results['video-id'], results['t-start'], results['t-end'], results['label'], results['score'])):
        t_start, t_end, label, score = t_start.item(), t_end.item(), label.item(), score.item()
        actionformer_res[video_id].append({
            'segment': [t_start, t_end], 'score': score, 'label': idx_classes[label]
        })
    save_obj = {"version": "1.0", "external_data": "", 'results': actionformer_res}
    with open('submission.json', 'w') as f:
        json.dump(save_obj, f)

    return 0



def valid_one_epoch_cl_single_gpu(
    val_qilDatasetList,
    model,
    curr_epoch,
    current_task_id,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
    logger=None,
    dataset_name='ego4d_cl',
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    total_R1_0_3 = AverageMeter()
    total_R5_0_3 = AverageMeter()
    total_R1_0_5 = AverageMeter()
    total_R5_0_5 = AverageMeter()
    total_mAP = AverageMeter()
    batch_time = AverageMeter()
    val_loaders_list = val_qilDatasetList.get_valSet_by_taskNum(current_task_id+1)
    # switch to evaluate mode
    results_list = []
    model.eval()
    if len(model.list_bias_layers) > 0:
        for bias_layer in model.list_bias_layers:
            bias_layer.eval()
    torch.set_grad_enabled(False)
    
    for n_task, (val_loader, num_queries) in enumerate(val_loaders_list):
        # dict for results (for our evaluation code)
        results = {
            'video-id': [],
            't-start' : [],
            't-end': [],
            'label': [],
            'score': []
        }

        # loop over validation set
        start = time.time()
        for iter_idx, video_list in enumerate(val_loader, 0):
            # forward the model (wo. grad)
            with torch.no_grad():

                output = model(video_list, task_id=current_task_id, is_training=False)
                # output, video_list, points, fpn_masks, out_cls_logits, out_offsets = model(video_list, hidden_state=True)
                # upack the results into ANet format
                # output = output1.detach().clone()
                # del output1
                # torch.cuda.empty_cache()
                num_vids = len(output)
                for vid_idx in range(num_vids):
                    if output[vid_idx]['segments'].shape[0] > 0:
                        results['video-id'].extend(
                            [output[vid_idx]['video_id']] *
                            output[vid_idx]['segments'].shape[0]
                        )
                        results['t-start'].append(output[vid_idx]['segments'][:, 0])
                        results['t-end'].append(output[vid_idx]['segments'][:, 1])
                        results['label'].append(output[vid_idx]['labels'])
                        results['score'].append(output[vid_idx]['scores'])

                # printing
                if (iter_idx != 0) and iter_idx % (print_freq) == 0:
                    # measure elapsed time (sync all kernels)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    batch_time.update((time.time() - start) / print_freq)
                    start = time.time()

                    # print timing
                    logger.info('Test: [{0:05d}/{1:05d}]\t'
                        'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                        iter_idx, len(val_loader), batch_time=batch_time))

        # gather all stats and evaluate
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()
        
        eval_result = None
        if dataset_name == "ego4d" or dataset_name == "ego4d_cl":
            #retrieval ######################### SAVE######################### SAVE######################### SAVE######################### SAVE######################### SAVE 
            classes = {'use_phone': 16, 'water_soil_/_plants_/_crops': 59, 'clean_/_wipe_a_table_or_kitchen_counter': 29, 'walk_down_stairs_/_walk_up_stairs': 30, 'arrange_/_organize_other_items': 24, 'clean_/_wipe_other_surface_or_object': 6, 'fill_a_pot_/_bottle_/_container_with_water': 64, 'use_a_laptop_/_computer': 37, 'knead_/_shape_/_roll-out_dough': 22, 'cut_dough': 25, 'fry_dough': 57, 'converse_/_interact_with_someone': 11, 'stir_/_mix_food_while_cooking': 21, 'wash_dishes_/_utensils_/_bakeware_etc.': 68, 'turn-on_/_light_the_stove_burner': 9, 'serve_food_onto_a_plate': 67, 'chop_/_cut_wood_pieces_using_tool': 82, 'cut_/_trim_grass_with_other_tools': 92, 'trim_hedges_or_branches': 100, 'browse_through_groceries_or_food_items_on_rack_/_shelf': 32, 'read_a_book_/_magazine_/_shopping_list_etc.': 28, 'take_photo_/_record_video_with_a_camera': 0, 'pay_at_billing_counter': 42, 'stand_in_the_queue_/_line_at_a_shop_/_supermarket': 53, 'browse_through_other_items_on_rack_/_shelf': 50, 'browse_through_clothing_items_on_rack_/_shelf_/_hanger': 2, 'look_at_clothes_in_the_mirror': 83, '"try-out_/_wear_accessories_(e.g._tie,_belt,_scarf)"': 102, 'put_away_(or_take_out)_dishes_/_utensils_in_storage': 81, 'clean_/_wipe_kitchen_appliance': 23, 'wash_vegetable_/_fruit_/_food_item': 95, '"cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat"': 75, 'cut_other_item_using_tool': 27, 'drill_into_wall_/_wood_/_floor_/_metal': 19, 'use_hammer_/_nail-gun_to_fix_nail': 34, 'weigh_food_/_ingredient_using_a_weighing_scale': 54, 'pack_food_items_/_groceries_into_bags_/_boxes': 41, 'drink_beverage': 65, 'withdraw_money_from_atm_/_operate_atm': 3, 'put_away_(or_take_out)_food_items_in_the_fridge': 39, 'interact_or_play_with_pet_/_animal': 101, 'put_away_(or_take_out)_ingredients_in_storage': 7, '"try-out_/_wear_clothing_items_(e.g._shirt,_jeans,_sweater)"': 77, 'throw_away_trash_/_put_trash_in_trash_can': 8, 'tie_up_branches_/_plants_with_string': 103, 'remove_weeds_from_ground': 85, 'collect_/_rake_dry_leaves_on_ground': 91, 'harvest_vegetables_/_fruits_/_crops_from_plants_on_the_ground': 86, 'place_items_in_shopping_cart': 31, 'write_notes_in_a_paper_/_book': 108, 'wash_hands': 5, 'pack_other_items_into_bags_/_boxes': 73, 'pack_soil_into_the_ground_or_a_pot_/_container': 47, 'plant_seeds_/_plants_/_flowers_into_ground': 48, '"level_ground_/_soil_(eg._using_rake,_shovel,_etc)"': 46, 'dig_or_till_the_soil_with_a_hoe_or_other_tool': 45, 'cut_tree_branch': 90, 'measure_wooden_item_using_tape_/_ruler': 35, 'mark_item_with_pencil_/_pen_/_marker': 36, 'compare_two_clothing_items': 97, 'do_some_exercise': 80, 'watch_television': 17, 'taste_food_while_cooking': 96, 'rinse_/_drain_other_food_item_in_sieve_/_colander': 71, 'use_a_vacuum_cleaner_to_clean': 15, 'fix_other_item': 20, 'smooth_wood_using_sandpaper_/_sander_/_tool': 88, 'dig_or_till_the_soil_by_hand': 98, 'hang_clothes_in_closet_/_on_hangers': 1, 'clean_/_wipe_/_oil_metallic_item': 72, 'fix_bonnet_/_engine_of_car': 107, 'hang_clothes_to_dry': 109, 'cut_/_trim_grass_with_a_lawnmower': 76, 'fold_clothes_/_sheets': 56, 'dismantle_other_item': 18, 'fix_/_remove_/_replace_a_tire_or_wheel': 84, 'move_/_shift_/_arrange_small_tools': 78, 'make_coffee_or_tea_/_use_a_coffee_machine': 63, 'play_board_game_or_card_game': 60, 'count_money_before_paying': 40, 'enter_a_supermarket_/_shop': 49, 'exit_a_supermarket_/_shop': 51, 'play_a_video_game': 79, 'arrange_pillows_on_couch_/_chair': 104, '"make_the_bed_/_arrange_pillows,_sheets_etc._on_bed"': 105, 'clean_/_sweep_floor_with_broom': 61, 'arrange_/_organize_clothes_in_closet/dresser': 55, 'load_/_unload_a_washing_machine_or_dryer': 89, 'move_/_shift_around_construction_material': 70, '"put_on_safety_equipment_(e.g._gloves,_helmet,_safety_goggles)"': 52, 'cut_open_a_package_(e.g._with_scissors)': 66, 'stir_/_mix_ingredients_in_a_bowl_or_pan_(before_cooking)': 4, 'fry_other_food_item': 38, 'eat_a_snack': 62, 'drive_a_vehicle': 99, 'arrange_/_organize_items_in_fridge': 10, 'browse_through_accessories_on_rack_/_shelf': 43, 'fix_wiring': 26, 'prepare_or_apply_cement_/_concrete_/_mortar': 69, 'put_food_into_the_oven_to_bake': 106, 'peel_a_fruit_or_vegetable': 74, 'smoke_cigar_/_cigarette_/_vape': 93, 'paint_using_paint_brush_/_roller': 14, 'climb_up_/_down_a_ladder': 12, 'cut_thread_/_paper_/_cardboard_using_scissors_/_knife_/_cutter': 44, 'plaster_wall_/_surface': 13, 'fix_pipe_/_plumbing': 87, '"clean_/_repair_small_equipment_(mower,_trimmer_etc.)"': 33, 'remove_food_from_the_oven': 58, 'iron_clothes_or_sheets': 94}
            # classes = {'clean_/_wipe_other_surface_or_object': 0, 'throw_away_trash_/_put_trash_in_trash_can': 1, 'walk_down_stairs_/_walk_up_stairs': 2, 'look_at_clothes_in_the_mirror': 3, 'put_away_(or_take_out)_food_items_in_the_fridge': 4, '"cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat"': 5, 'read_a_book_/_magazine_/_shopping_list_etc.': 6, 'browse_through_clothing_items_on_rack_/_shelf_/_hanger': 7, 'use_phone': 8, 'converse_/_interact_with_someone': 9}
            idx_classes = {}
            # classes_per_task = 10
            # classes_per_task = 1
            # max_classes = (n_task + 1) * classes_per_task
            for key, value in classes.items():
                idx_classes[value] = key
            from collections import defaultdict
            from tqdm import tqdm
            import json

            # # test upperbound for cls
            actionformer_res = defaultdict(list)
            
            for video_id, t_start, t_end, label, score in tqdm(zip(results['video-id'], results['t-start'], results['t-end'], results['label'], results['score'])):
                # test upperbound for cls
                t_start, t_end, label, score = t_start.item(), t_end.item(), label.item(), score.item()
                actionformer_res[video_id].append({
                    'segment': [t_start, t_end], 'score': score, 'label': idx_classes[label]
                })
            save_obj = {"version": "1.0", "external_data": "", 'results': actionformer_res}
            
            if not os.path.exists("retrieval_json"):
                os.makedirs("retrieval_json")
            now = time.time() 
            now = str(int(round(now * 1000)))
            pred_file = os.path.join("retrieval_json", f"results_{output_file}_{now}.json")

            with open(pred_file, 'w') as f:
                json.dump(save_obj, f)
            eval_result = evaluation_retrieval(gt="data/ego4d/ego4d_mq_query_incremental_22_all.pkl",
                                pred=f"{pred_file}",
                                subset="val",
                                tiou=[0.1, 0.2, 0.3, 0.4, 0.5],
                                use_cl=True,
                                current_task_id=n_task,
                                )
            tious = [0.1, 0.2, 0.3, 0.4, 0.5]
            recalls = [1, 5]
            for i, t in enumerate(tious):
                for j, r in enumerate(recalls):
                    recall = eval_result[i, j]
                    logger.info(f'Task {n_task} Rank {r}x @ tIoU {t} is {recall}')
            os.remove(pred_file)
            ######################### SAVE######################### SAVE######################### SAVE######################### SAVE######################### SAVE

        assert evaluator is not None
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        mAP, avg_mAP, tiou_thresholds = evaluator.evaluate(results, current_task_id=current_task_id, verbose=False)
        for tiou, tiou_mAP in zip(tiou_thresholds, mAP):
            logger.info(f'Task {n_task} tIoU = {tiou:.1f}: mAP = {tiou_mAP * 100:.2f} %')
        logger.info(f'Task {n_task} Average Map is :{avg_mAP * 100: .2f} %')
    
        total_R1_0_3.update(eval_result[2, 0], num_queries)
        total_R5_0_3.update(eval_result[2, 1], num_queries)
        total_R1_0_5.update(eval_result[4, 0], num_queries)
        total_R5_0_5.update(eval_result[4, 1], num_queries)
        total_mAP.update(avg_mAP, num_queries)
    
    logger.info(f'Pre Testing Results: Averge Rank 1x @ tIoU 0.3 is {total_R1_0_3.avg}')
    logger.info(f'Averge Rank 5x @ tIoU 0.3 is {total_R5_0_3.avg}')
    logger.info(f'Averge Rank 1x @ tIoU 0.5 is {total_R1_0_5.avg}')
    logger.info(f'Averge Rank 5x @ tIoU 0.5 is {total_R5_0_5.avg}')
    logger.info(f'Average Map is :{total_mAP.avg * 100: .2f} %')

    return total_R1_0_3.avg, total_R5_0_3.avg, total_R1_0_5.avg, total_R5_0_5.avg, total_mAP.avg


def final_validate(
    val_qilDatasetList,
    model,
    curr_epoch,
    current_task_id,
    ext_score_file = None,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
    logger=None,
    dataset_name = 'ego4d_cl',
    list_val_recall_ii = {'val': []},
    list_val_mAP_ii = {'val': []},
    type_val = 'val',
):
    """Test the model on the validation set"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    total_R1_0_3 = AverageMeter()
    total_R5_0_3 = AverageMeter()
    total_R1_0_5 = AverageMeter()
    total_R5_0_5 = AverageMeter()
    total_mAP = AverageMeter()
    BWF_R1_0_5 = AverageMeter()
    BWF_mAP = AverageMeter()
    batch_time = AverageMeter()
    val_loaders_list = val_qilDatasetList.get_valSet_by_taskNum(current_task_id+1)
    # switch to evaluate mode
    results_list = []
    model.eval()
    if len(model.list_bias_layers) > 0:
        for bias_layer in model.list_bias_layers:
            bias_layer.eval()
    torch.set_grad_enabled(False)
    
    for n_task, (val_loader, num_queries) in enumerate(val_loaders_list):
        # dict for results (for our evaluation code)
        results = {
            'video-id': [],
            't-start' : [],
            't-end': [],
            'label': [],
            'score': []
        }

        # loop over validation set
        start = time.time()
        for iter_idx, video_list in enumerate(val_loader, 0):
            # forward the model (wo. grad)
            with torch.no_grad():

                output = model(video_list, task_id=current_task_id, is_training=False, val_qilDatasetList=val_qilDatasetList)
                # output, video_list, points, fpn_masks, out_cls_logits, out_offsets = model(video_list, hidden_state=True)
                # upack the results into ANet format
                # output = output1.detach().clone()
                # del output1
                # torch.cuda.empty_cache()
                num_vids = len(output)
                for vid_idx in range(num_vids):
                    if output[vid_idx]['segments'].shape[0] > 0:
                        results['video-id'].extend(
                            [output[vid_idx]['video_id']] *
                            output[vid_idx]['segments'].shape[0]
                        )
                        results['t-start'].append(output[vid_idx]['segments'][:, 0])
                        results['t-end'].append(output[vid_idx]['segments'][:, 1])
                        results['label'].append(output[vid_idx]['labels'])
                        results['score'].append(output[vid_idx]['scores'])

                # printing
                if (iter_idx != 0) and iter_idx % (print_freq) == 0:
                    # measure elapsed time (sync all kernels)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    batch_time.update((time.time() - start) / print_freq)
                    start = time.time()

                    # print timing
                    logger.info('Test: [{0:05d}/{1:05d}]\t'
                        'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                        iter_idx, len(val_loader), batch_time=batch_time))

        # gather all stats and evaluate
        results['t-start'] = torch.cat(results['t-start']).numpy()
        results['t-end'] = torch.cat(results['t-end']).numpy()
        results['label'] = torch.cat(results['label']).numpy()
        results['score'] = torch.cat(results['score']).numpy()
        
        eval_result = None
        if dataset_name == "ego4d" or dataset_name == "ego4d_cl":
            #retrieval ######################### SAVE######################### SAVE######################### SAVE######################### SAVE######################### SAVE 
            classes = {'use_phone': 16, 'water_soil_/_plants_/_crops': 59, 'clean_/_wipe_a_table_or_kitchen_counter': 29, 'walk_down_stairs_/_walk_up_stairs': 30, 'arrange_/_organize_other_items': 24, 'clean_/_wipe_other_surface_or_object': 6, 'fill_a_pot_/_bottle_/_container_with_water': 64, 'use_a_laptop_/_computer': 37, 'knead_/_shape_/_roll-out_dough': 22, 'cut_dough': 25, 'fry_dough': 57, 'converse_/_interact_with_someone': 11, 'stir_/_mix_food_while_cooking': 21, 'wash_dishes_/_utensils_/_bakeware_etc.': 68, 'turn-on_/_light_the_stove_burner': 9, 'serve_food_onto_a_plate': 67, 'chop_/_cut_wood_pieces_using_tool': 82, 'cut_/_trim_grass_with_other_tools': 92, 'trim_hedges_or_branches': 100, 'browse_through_groceries_or_food_items_on_rack_/_shelf': 32, 'read_a_book_/_magazine_/_shopping_list_etc.': 28, 'take_photo_/_record_video_with_a_camera': 0, 'pay_at_billing_counter': 42, 'stand_in_the_queue_/_line_at_a_shop_/_supermarket': 53, 'browse_through_other_items_on_rack_/_shelf': 50, 'browse_through_clothing_items_on_rack_/_shelf_/_hanger': 2, 'look_at_clothes_in_the_mirror': 83, '"try-out_/_wear_accessories_(e.g._tie,_belt,_scarf)"': 102, 'put_away_(or_take_out)_dishes_/_utensils_in_storage': 81, 'clean_/_wipe_kitchen_appliance': 23, 'wash_vegetable_/_fruit_/_food_item': 95, '"cut_/_chop_/_slice_a_vegetable,_fruit,_or_meat"': 75, 'cut_other_item_using_tool': 27, 'drill_into_wall_/_wood_/_floor_/_metal': 19, 'use_hammer_/_nail-gun_to_fix_nail': 34, 'weigh_food_/_ingredient_using_a_weighing_scale': 54, 'pack_food_items_/_groceries_into_bags_/_boxes': 41, 'drink_beverage': 65, 'withdraw_money_from_atm_/_operate_atm': 3, 'put_away_(or_take_out)_food_items_in_the_fridge': 39, 'interact_or_play_with_pet_/_animal': 101, 'put_away_(or_take_out)_ingredients_in_storage': 7, '"try-out_/_wear_clothing_items_(e.g._shirt,_jeans,_sweater)"': 77, 'throw_away_trash_/_put_trash_in_trash_can': 8, 'tie_up_branches_/_plants_with_string': 103, 'remove_weeds_from_ground': 85, 'collect_/_rake_dry_leaves_on_ground': 91, 'harvest_vegetables_/_fruits_/_crops_from_plants_on_the_ground': 86, 'place_items_in_shopping_cart': 31, 'write_notes_in_a_paper_/_book': 108, 'wash_hands': 5, 'pack_other_items_into_bags_/_boxes': 73, 'pack_soil_into_the_ground_or_a_pot_/_container': 47, 'plant_seeds_/_plants_/_flowers_into_ground': 48, '"level_ground_/_soil_(eg._using_rake,_shovel,_etc)"': 46, 'dig_or_till_the_soil_with_a_hoe_or_other_tool': 45, 'cut_tree_branch': 90, 'measure_wooden_item_using_tape_/_ruler': 35, 'mark_item_with_pencil_/_pen_/_marker': 36, 'compare_two_clothing_items': 97, 'do_some_exercise': 80, 'watch_television': 17, 'taste_food_while_cooking': 96, 'rinse_/_drain_other_food_item_in_sieve_/_colander': 71, 'use_a_vacuum_cleaner_to_clean': 15, 'fix_other_item': 20, 'smooth_wood_using_sandpaper_/_sander_/_tool': 88, 'dig_or_till_the_soil_by_hand': 98, 'hang_clothes_in_closet_/_on_hangers': 1, 'clean_/_wipe_/_oil_metallic_item': 72, 'fix_bonnet_/_engine_of_car': 107, 'hang_clothes_to_dry': 109, 'cut_/_trim_grass_with_a_lawnmower': 76, 'fold_clothes_/_sheets': 56, 'dismantle_other_item': 18, 'fix_/_remove_/_replace_a_tire_or_wheel': 84, 'move_/_shift_/_arrange_small_tools': 78, 'make_coffee_or_tea_/_use_a_coffee_machine': 63, 'play_board_game_or_card_game': 60, 'count_money_before_paying': 40, 'enter_a_supermarket_/_shop': 49, 'exit_a_supermarket_/_shop': 51, 'play_a_video_game': 79, 'arrange_pillows_on_couch_/_chair': 104, '"make_the_bed_/_arrange_pillows,_sheets_etc._on_bed"': 105, 'clean_/_sweep_floor_with_broom': 61, 'arrange_/_organize_clothes_in_closet/dresser': 55, 'load_/_unload_a_washing_machine_or_dryer': 89, 'move_/_shift_around_construction_material': 70, '"put_on_safety_equipment_(e.g._gloves,_helmet,_safety_goggles)"': 52, 'cut_open_a_package_(e.g._with_scissors)': 66, 'stir_/_mix_ingredients_in_a_bowl_or_pan_(before_cooking)': 4, 'fry_other_food_item': 38, 'eat_a_snack': 62, 'drive_a_vehicle': 99, 'arrange_/_organize_items_in_fridge': 10, 'browse_through_accessories_on_rack_/_shelf': 43, 'fix_wiring': 26, 'prepare_or_apply_cement_/_concrete_/_mortar': 69, 'put_food_into_the_oven_to_bake': 106, 'peel_a_fruit_or_vegetable': 74, 'smoke_cigar_/_cigarette_/_vape': 93, 'paint_using_paint_brush_/_roller': 14, 'climb_up_/_down_a_ladder': 12, 'cut_thread_/_paper_/_cardboard_using_scissors_/_knife_/_cutter': 44, 'plaster_wall_/_surface': 13, 'fix_pipe_/_plumbing': 87, '"clean_/_repair_small_equipment_(mower,_trimmer_etc.)"': 33, 'remove_food_from_the_oven': 58, 'iron_clothes_or_sheets': 94}
            idx_classes = {}
            # classes_per_task = 10
            # max_classes = (n_task + 1) * classes_per_task
            for key, value in classes.items():
                idx_classes[value] = key
            from collections import defaultdict
            from tqdm import tqdm
            import json

            # # test upperbound for cls
            actionformer_res = defaultdict(list)
            
            for video_id, t_start, t_end, label, score in tqdm(zip(results['video-id'], results['t-start'], results['t-end'], results['label'], results['score'])):
                # test upperbound for cls
                t_start, t_end, label, score = t_start.item(), t_end.item(), label.item(), score.item()
                actionformer_res[video_id].append({
                    'segment': [t_start, t_end], 'score': score, 'label': idx_classes[label]
                })
            save_obj = {"version": "1.0", "external_data": "", 'results': actionformer_res}
            
            if not os.path.exists("retrieval_json"):
                os.makedirs("retrieval_json")
            now = time.time() 
            now = str(int(round(now * 1000)))
            pred_file = os.path.join("retrieval_json", f"results_{output_file}_{now}.json")

            with open(pred_file, 'w') as f:
                json.dump(save_obj, f)
            eval_result = evaluation_retrieval(gt="data/ego4d/ego4d_mq_query_incremental_22_all.pkl",
                                pred=f"{pred_file}",
                                subset="val",
                                tiou=[0.1, 0.2, 0.3, 0.4, 0.5],
                                use_cl=True,
                                current_task_id=n_task,
                                )
            tious = [0.1, 0.2, 0.3, 0.4, 0.5]
            recalls = [1, 5]
            for i, t in enumerate(tious):
                for j, r in enumerate(recalls):
                    recall = eval_result[i, j]
                    logger.info(f'Task {n_task} Rank {r}x @ tIoU {t} is {recall}')
            os.remove(pred_file)
            ######################### SAVE######################### SAVE######################### SAVE######################### SAVE######################### SAVE

        assert evaluator is not None
        if (ext_score_file is not None) and isinstance(ext_score_file, str):
            results = postprocess_results(results, ext_score_file)
        # call the evaluator
        mAP, avg_mAP, tiou_thresholds = evaluator.evaluate(results, current_task_id=current_task_id, verbose=False)
        for tiou, tiou_mAP in zip(tiou_thresholds, mAP):
            logger.info(f'Task {n_task} tIoU = {tiou:.1f}: mAP = {tiou_mAP * 100:.2f} %')
        logger.info(f'Task {n_task} Average Map is :{avg_mAP * 100: .2f} %')
    
        total_R1_0_3.update(eval_result[2, 0], num_queries)
        total_R5_0_3.update(eval_result[2, 1], num_queries)
        total_R1_0_5.update(eval_result[4, 0], num_queries)
        total_R5_0_5.update(eval_result[4, 1], num_queries)
        total_mAP.update(avg_mAP, num_queries)
        
        if n_task == current_task_id:
            list_val_recall_ii[type_val].append(eval_result[4, 0])
        elif n_task < current_task_id:
            forgetting_recall = list_val_recall_ii[type_val][n_task] - eval_result[4, 0]
            BWF_R1_0_5.update(forgetting_recall, num_queries)
            
        if n_task == current_task_id:
            list_val_mAP_ii[type_val].append(avg_mAP)
        elif n_task < current_task_id:
            forgetting_mAP = list_val_mAP_ii[type_val][n_task] - avg_mAP
            BWF_mAP.update(forgetting_mAP, num_queries)
            
    
    logger.info(f'Pre Testing Results: Averge Rank 1x @ tIoU 0.3 is {total_R1_0_3.avg}')
    logger.info(f'Averge Rank 5x @ tIoU 0.3 is {total_R5_0_3.avg}')
    logger.info(f'Averge Rank 1x @ tIoU 0.5 is {total_R1_0_5.avg}')
    logger.info(f'Averge Rank 5x @ tIoU 0.5 is {total_R5_0_5.avg}')
    logger.info(f'Average Map is :{total_mAP.avg * 100: .2f} %')
    logger.info(f'Total BWF Rank 1x @ @ tIoU 0.5 per task: {BWF_R1_0_5.avg}')
    logger.info(f'Total BWF mAP per task: {BWF_mAP.avg}')

    return total_R1_0_3.avg, total_R5_0_3.avg, total_R1_0_5.avg, total_R5_0_5.avg, total_mAP.avg, BWF_R1_0_5.avg, BWF_mAP.avg