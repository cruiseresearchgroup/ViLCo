import torch.nn.functional as F
import torch
from utils import exp_utils, train_utils, loss_utils, vis_utils
from dataset import dataset_utils

# return the regularized loss
def get_mas_regularized_loss(loss, model, reg_lambda):
    mas_reg = model.module.reg_params
    if 'importance' in mas_reg and 'optpar' in mas_reg:
        importance_dict_list = mas_reg['importance']
        optpar_dict_list = mas_reg['optpar']

        for i in range(len(importance_dict_list)):
            for name, param in model.module.named_parameters():
                if 'scale' not in name and (name in importance_dict_list[i].keys()):
                    importance = importance_dict_list[i][name]
                    optpar = optpar_dict_list[i][name]
                    if optpar.size(0) == param.size(0):
                        loss = loss + (importance * (optpar - param).pow(2)).sum() * reg_lambda
                    else:
                        size_optpar = optpar.size(0)
                        loss = loss + (importance * (optpar - param[:size_optpar]).pow(2)).sum() * reg_lambda
    return loss

def on_task_mas_update(loader_task, device, optimizer, model, config, current_task_id):
    
    print('MAS HERE')
    model.train()
    optimizer.zero_grad()
    
    mas_reg = model.module.reg_params
    
    if 'importance' in mas_reg and 'optpar' in mas_reg:
        importance_dict_list = mas_reg['importance']
        optpar_dict_list = mas_reg['optpar']
    else:
        mas_reg['importance'] = []
        mas_reg['optpar'] = []

    for batch_idx, sample in enumerate(loader_task):
        optimizer.zero_grad(set_to_none=True)
        iter_num = batch_idx
        sample = exp_utils.dict_to_cuda(sample)
        sample = dataset_utils.process_data(config, sample, iter=iter_num, split='train', device=device)
        clips, queries = sample['clip'], sample['query']
        preds = model(clips, queries, training=True, fix_backbone=config.model.fix_backbone, task_id=current_task_id)
        losses, preds_top, sample = loss_utils.get_losses_with_anchor(config, preds, sample)
        loss = loss['final_loss']
        total_loss = 0.0
        for k, v in losses.items():
            if 'loss' in k:
                total_loss += losses[k.replace('loss_', 'weight_')] * v
        total_loss = total_loss / config.train.accumulation_step
        total_loss.backward()
        
    importance_dict = {}
    optpar_dict = {}

    # gradients accumulated can be used to calculate importance
    for name, param in model.module.named_parameters():
        if param.grad is not None:
            optpar_dict[name] = param.data.clone()
            importance_dict[name] = param.grad.data.clone().abs() # param.grad.data.clone().pow(2)
        
    mas_reg['importance'].append(importance_dict)
    mas_reg['optpar'].append(optpar_dict)
    
    return mas_reg