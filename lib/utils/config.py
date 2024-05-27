# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os

from yacs.config import CfgNode as CN

_C = CN(new_allowed=True)

# needed by trainer
_C.name = "default"
_C.dataset = CN()
_C.test = CN()


def get_cfg_defaults(cfg_file):
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    _C = CN(new_allowed=True)
    return update_cfg(_C, cfg_file)


# Alternatively, provide a way to import the defaults as
# a global singleton:
cfg = _C    # users can `from config import cfg`

# cfg = get_cfg_defaults()
# cfg.merge_from_file('./configs/example.yaml')

# # Now override from a list (opts could come from the command line)
# opts = ['dataset.root', './data/XXXX', 'learning_rate', '1e-2']
# cfg.merge_from_list(opts)


def update_cfg(cfg, cfg_file):
    # cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    # return cfg.clone()
    return cfg


def parse_args(args):
    cfg_file = args.cfg_file
    if args.cfg_file is not None:
        cfg = update_cfg(args.cfg_file)
    else:
        cfg = get_cfg_defaults()

    # if args.misc is not None:
    #     cfg.merge_from_list(args.misc)

    return cfg


def parse_args_extend(args):
    if args.resume:
        if not os.path.exists(args.log_dir):
            raise ValueError("Experiment are set to resume mode, but log directory does not exist.")

        # load log's cfg
        cfg_file = os.path.join(args.log_dir, "cfg.yaml")
        cfg = update_cfg(cfg_file)

        if args.misc is not None:
            cfg.merge_from_list(args.misc)
    else:
        parse_args(args)



