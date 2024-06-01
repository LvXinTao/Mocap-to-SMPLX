import torch

def init_params(params,body_models,nf):
    """
    Using pose mean to initialize params,and move to cuda
    """
    for key in params.keys():
        params[key]=torch.tensor(params[key],dtype=torch.float32,
                                 device=torch.device('cuda'))
    params['left_hand_pose']=body_models.left_hand_mean\
                            .reshape((15,3))[None,...].expand([nf,-1,-1]).clone().to('cuda')
    params['right_hand_pose']=body_models.right_hand_mean\
                            .reshape((15,3))[None,...].expand([nf,-1,-1]).clone().to('cuda')
    return params

def tensor_to_numpy(params):
    for key in params.keys():
        params[key]=params[key].detach().cpu().numpy()
    return params

def numpy_to_tensor(params):
    for key in params.keys():
        params[key]=torch.tensor(params[key],dtype=torch.float32,
                                 device=torch.device('cuda'))
    return params

def npz_to_dict(npz):
    new_dict={}
    for key in npz.files:
        new_dict[key]=npz[key]
    return new_dict
