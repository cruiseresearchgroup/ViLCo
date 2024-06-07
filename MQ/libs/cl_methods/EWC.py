import torch.nn.functional as F
import tqdm
from torch.autograd import Variable
import torch

def get_regularized_loss(loss, model, ewc_lambda):
    ewc_reg = model.reg_params
    if 'fisher' in ewc_reg and 'optpar' in ewc_reg:
        fisher_dict_list = ewc_reg['fisher']
        optpar_dict_list = ewc_reg['optpar']

        for i in range(len(fisher_dict_list)):
            for name, param in model.named_parameters():
                if 'scale' not in name and (name in fisher_dict_list[i].keys()):
                    fisher = fisher_dict_list[i][name]
                    optpar = optpar_dict_list[i][name]
                    if optpar.size(0) == param.size(0):
                        loss = loss + (fisher * (optpar - param).pow(2)).sum() * ewc_lambda
                    else:
                        size_optpar = optpar.size(0)
                        loss = loss + (fisher * (optpar - param[:size_optpar]).pow(2)).sum() * ewc_lambda
    return loss

def on_task_update(loader_task, device, optimizer, model):
    
    print('EWC HERE')
    model.train()
    ewc_reg = model.reg_params
    
    if 'fisher' in ewc_reg and 'optpar' in ewc_reg:
        fisher_dict_list = ewc_reg['fisher']
        optpar_dict_list = ewc_reg['optpar']
    else:
        ewc_reg['fisher'] = []
        ewc_reg['optpar'] = []

    for iter_idx, video_list in enumerate(loader_task, 0):
        optimizer.zero_grad(set_to_none=True)
        loss = model(video_list)
        loss = loss['final_loss']
        # loss = Variable(loss['final_loss'], requires_grad=True)
        loss.backward()
        
    fisher_dict = {}
    optpar_dict = {}
    
    # gradients accumulated can be used to calculate fisher
    for name, param in model.named_parameters():
        if param.grad is not None:
            optpar_dict[name] = param.data.clone()
            fisher_dict[name] = param.grad.data.clone().pow(2)
        
    ewc_reg['fisher'].append(fisher_dict)
    ewc_reg['optpar'].append(optpar_dict)
    
    return ewc_reg