import os

def check_key(cfg, keys):
    if cfg is None:
        return False

    for key in keys:
        if key in cfg.keys():
            cfg = cfg[key]
        else:
            return False
    return cfg is not None

def check_files(files, lists):
    for file in lists:
        if file not in files.keys():
            raise Exception(f'Miss \'{file.upper()}\' file, please check the data root and config settings.')
        elif not os.path.exists(files[file]):
            raise Exception(f'Miss \'{file.upper()}\' file in \'{files[file]}\'.')
    return True

def update_cfg_inpaint(cfg_default, cfg):

    if check_key(cfg, ['parameters', 'inpaint']):
            for key in cfg.parameters.inpaint.keys():
                    cfg_default.test[key] = cfg.parameters.inpaint[key]
    
    return cfg_default

def update_cfg_refine(cfg_default, cfg):

    if check_key(cfg, ['parameters', 'refine']):
            for key in cfg.parameters.refine.keys():
                    cfg_default.test[key] = cfg.parameters.refine[key]
    
    return cfg_default
