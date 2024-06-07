import torch.nn.functional as F
import torch

# return the regularized loss
def get_mas_regularized_loss(loss, model, reg_lambda):
    mas_reg = model.reg_params
    if 'importance' in mas_reg and 'optpar' in mas_reg:
        importance_dict_list = mas_reg['importance']
        optpar_dict_list = mas_reg['optpar']

        for i in range(len(importance_dict_list)):
            for name, param in model.named_parameters():
                if 'scale' not in name and (name in importance_dict_list[i].keys()):
                    importance = importance_dict_list[i][name]
                    optpar = optpar_dict_list[i][name]
                    if optpar.size(0) == param.size(0):
                        loss = loss + (importance * (optpar - param).pow(2)).sum() * reg_lambda
                    else:
                        size_optpar = optpar.size(0)
                        loss = loss + (importance * (optpar - param[:size_optpar]).pow(2)).sum() * reg_lambda
    return loss

def on_task_mas_update(loader_task, device, optimizer, model):
    
    print('MAS HERE')
    model.train()
    optimizer.zero_grad()
    
    mas_reg = model.reg_params
    
    if 'importance' in mas_reg and 'optpar' in mas_reg:
        importance_dict_list = mas_reg['importance']
        optpar_dict_list = mas_reg['optpar']
    else:
        mas_reg['importance'] = []
        mas_reg['optpar'] = []

    for iter_idx, video_list in enumerate(loader_task, 0):
        optimizer.zero_grad(set_to_none=True)
        loss = model(video_list)
        loss = loss['final_loss']
        # loss = Variable(loss['final_loss'], requires_grad=True)
        loss.backward()
        
    importance_dict = {}
    optpar_dict = {}

    # gradients accumulated can be used to calculate importance
    for name, param in model.named_parameters():
        if param.grad is not None:
            optpar_dict[name] = param.data.clone()
            importance_dict[name] = param.grad.data.clone().abs() # param.grad.data.clone().pow(2)
        
    mas_reg['importance'].append(importance_dict)
    mas_reg['optpar'].append(optpar_dict)
    
    return mas_reg

# def consolidate_reg_params(model):
#     """
#     Input:
#     1) model: A reference to the model that is being trained
#     Output:
#     1) reg_params: A dictionary containing importance weights (importance), init_val (keep a reference 
#     to the initial values of the parameters) for all trainable parameters
#     Function: This function updates the value (adds the value) of omega across the tasks that the model is 
#     exposed to
    
#     """
#     #Get the reg_params for the model 
#     reg_params = model.reg_params

#     for name, param in model.tmodel.named_parameters():
#         if param in reg_params:
#             param_dict = reg_params[param]
#             print ("Consolidating the omega values for layer", name)
            
#             #Store the previous values of omega
#             prev_omega = param_dict['prev_omega']
#             new_omega = param_dict['importance']

#             new_omega = torch.add(prev_omega, new_omega)
#             del param_dict['prev_omega']
            
#             param_dict['importance'] = new_omega

#             #the key for this dictionary is the name of the layer
#             reg_params[param] = param_dict

#     model.reg_params = reg_params

#     return model