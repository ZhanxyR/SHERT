import torch
import torch.distributed as dist
import logging

def load_model(model, optimizer, scheduler, ckpt_path, device):
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state_dict['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    if scheduler is not None:
        scheduler.load_state_dict(state_dict['scheduler'])
    logging.info(f'Model loaded from {ckpt_path} on {device}')
    start_epoch = state_dict['epoch'] + 1
    return start_epoch

def save_model(epoch, model, optimizer, scheduler, ckpt_path):
    state = {}
    state['model'] = model.module.state_dict() if dist.is_initialized() else model.state_dict()
    state['optimizer'] = optimizer.state_dict()
    state['scheduler'] = scheduler.state_dict()
    state['epoch'] = epoch
    torch.save(state, ckpt_path)

def reduce_tensor(tensor: torch.Tensor):
    value = tensor.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value