import numpy as np
import logging
import torch
import argparse
import os
from lib.utils.common_util import check_key, check_files, update_cfg_refine, update_cfg_inpaint
from lib.utils.config import  get_cfg_defaults
from lib.utils.uv_sample.divided_uv_generator import Index_UV_Generator
from lib.tools.subdivide_smplx import subdivide
from lib.tools.semantic_normal_sampling import Semantic_Normal_Sampling
from lib.tools.gen_smplx import gen_star_smplx
from lib.tools.texture_projection import texture_projection
from lib.tools.face_substitution import face_smooth, face_substitution
from lib.datasets.InpaintEvalDataset import InapintEvalDataset
from lib.datasets.RefineEvalDataset import RefineEvalDataset
from lib.models.InpaintUNet import InpaintUNet
from lib.models.RefineUNet import RefineUNet
from apps.test_inpaint import evaluate_inpaint
from apps.test_refine import evaluate_refine

def parse():

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gpu', type=int, default=0, help='The GPU device to be used.')
    parser.add_argument('-e', '--example', type=str, default='image_w_gt', choices=['scan', 'image', 'image_w_gt'])
    parser.add_argument('-i', '--input', type=str, default=None, help='The root path used for files loading.')
    parser.add_argument('-o', '--output', type=str, default=None, help='The folder used for output. Default as \'$root$/results\'.')
    parser.add_argument('-c', '--config', type=str, default='config.yaml', help='The config file name in root.')
    parser.add_argument('-r', '--resources', type=str, default='./lib/configs/resources.yaml', help='The resources file path.')

    return parser.parse_args()

def load_config(args):

    if args.input is not None:
        # User defined
        cfg_path = os.path.join(args.input, args.config)
        if not os.path.exists(cfg_path):
            raise Exception(f'Can not find \'{args.config}\' in given root \'{args.input}\'.')
    elif args.example == 'image_w_gt':
        # Use image-pred mesh and fitted smplx.
        cfg_path = './examples/demo_image_w_gt_smplx/config.yaml'
    elif args.example == 'scan':
        # Use THuman scan and fitted smplx.
        cfg_path = './examples/demo_scan/config.yaml'
    elif args.example == 'image':
        # Given only image and predict all inputs with ECON.
        cfg_path = './examples/demo_image/config.yaml'
    else:
        # TODO: more, batch
        raise Exception(f'Not ready for example \'{args.example}\'. Coming soon.')

    cfg = get_cfg_defaults(cfg_path)
    logging.info(f'Load configs from \'{cfg_path}\'.')

    return cfg, cfg_path

if __name__ == '__main__':

    logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)

    args = parse()

    try:

        cfg, cfg_path = load_config(args)
        cfg_resources = get_cfg_defaults(args.resources)
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

        # Define the data root
        if args.input is not None:
            root = args.input
        elif check_key(cfg, ['root']):
            root = cfg.root
        else:
            raise Exception(f'Please define the data root with \'-r\' or modify the config file.')
        logging.info(f'Load files from \'{root}\'.')

        # Define the save dir
        if args.output is not None:
            save_root = args.output
        else:
            save_root = os.path.join(root, 'results')
        logging.info(f'Save results to \'{save_root}\'.')

        # Load files from data root
        files = {}
        # files = load_files(root, cfg)
        if check_key(cfg, ['files']):
            for key in cfg.files.keys():
                if cfg.files[key] is not None:
                    path = os.path.join(root, cfg.files[key])
                    if os.path.exists(path):
                        files[key] = path
                    else:
                        logging.warning(f'\'{key}\' in \'{path}\' is not exist.')
                else:
                    logging.warning(f'Empety \'{key.upper()}\' in \'{cfg_path}\'.')

        # Predict the target mesh from given image using ECON
        if check_key(cfg, ['settings', 'use_econ']) and cfg.settings.use_econ:
            # TODO: ECON pred

            # cfg.parameters['use_thuman_smplx_param'] = False
            # cfg.parameters['normal_flip'] = False

            raise Exception('TODO')

        '''
        ##############################
                    SNS
        ##############################
        '''
        '''
        Subdivision
        '''
        if not (check_key(cfg, ['settings', 'subdivide']) and not cfg.settings.subdivide):
            # used inputs
            check_files(files, ['smplx'])
            logging.info(f'Start \'Subdivision\'.')

            # process
            _, _, smplx_d2_path = subdivide(cfg_resources.models.smplx_eyes, files['smplx'], save_root=save_root)

            files['subsmplx'] = smplx_d2_path
            # check_files(files, ['smplx'])
        else:
            logging.info(f'Skip \'Subdivision\'.')

        # Prepare
        uv_sampler = Index_UV_Generator(data_dir=cfg_resources.others.smplx_official_template_root).to(device)
        sns_sampler = Semantic_Normal_Sampling(cfg_resources, uv_sampler, device, cfg_parameters=cfg)

        '''
        Sampling
        '''
        if not (check_key(cfg, ['settings', 'sns']) and not cfg.settings.sns):
            # used inputs
            check_files(files, ['mesh', 'subsmplx', 'smplx_param'])
            logging.info(f'Start \'Semantic- and Normal-based Sampling\'.')

            # process
            mesh_path, error_uv_path, error_index_path = sns_sampler.sample(files['mesh'], files['subsmplx'], files['smplx_param'], save_root=save_root)

            files['sampled_mesh'] = mesh_path
            files['error_uv'] = error_uv_path
            files['error_index'] = error_index_path

        else:
            logging.info(f'Skip \'Semantic- and Normal-based Sampling\'.')

        '''
        ##############################
                Completion
        ##############################
        '''

        # TODO: complete without smplx params

        '''
        Gen smplx_star
        '''
        if not check_key(cfg, ['files', 'smplx_star']):
            logging.info(f'Gen \'Star-SMPLX\'.')
            smplx_star_path = gen_star_smplx(files['smplx_param'], save_root, cfg_resources) 
            files['smplx_star'] = smplx_star_path
        else:
            # Define in the config.yaml to skip
            logging.info(f'Use given \'Star-SMPLX\'.')

        '''
        Completion
        '''
        if not (check_key(cfg, ['settings', 'complete']) and not cfg.settings.complete): 
            # used inputs
            check_files(files, ['sampled_mesh', 'subsmplx', 'error_uv', 'smplx_param', 'smplx_star'])
            logging.info(f'Start \'Completion\'.')

            # init model
            torch.cuda.empty_cache()
            cfg_inpaint = get_cfg_defaults(cfg_resources.configs.default_inpaint)
            cfg_inpaint = update_cfg_inpaint(cfg_inpaint, cfg)
            model = InpaintUNet(cfg_inpaint.model)
            model.requires_grad_(False)
            model.eval()
            model.cuda()
            
            # load ckpt
            state_dict = torch.load(cfg_inpaint.test.ckpt_path, map_location=device)
            model.load_state_dict(state_dict['model'])
            logging.info(f'Load ckpt from \'{cfg_inpaint.test.ckpt_path}\' on \'{device}\'') 

            # process
            # TODO: add dilation
            test_dataset = InapintEvalDataset([files], cfg_inpaint.test, cfg_resources)
            _, _, _, completed_mesh = evaluate_inpaint(uv_sampler, model, test_dataset, cfg_inpaint, cfg_resources, device, save_root=save_root)

            files['completed_mesh'] = completed_mesh

        else:
            logging.info(f'Skip \'Completion\'.')

        '''
        Face substitution with EMOCA
        '''
        if not (check_key(cfg, ['settings', 'use_emoca_face']) and not cfg.settings.use_emoca_face):
            if not check_key(cfg, ['files', 'emoca_face']):
                # Predict detailed face with EMOCA
                # TODO:
                logging.warning('Ignore! Please provide EMOCA result manually. Example in \'./examples/demo_image_w_gt_smplx/emoca_face.obj\'')
            else:
                # used inputs
                check_files(files, ['completed_mesh', 'subsmplx', 'emoca_face'])
                logging.info('Substitute with detailed face.')

                face_substitution_path = face_substitution(files, cfg_resources, uv_sampler, device, save_root=save_root)

                files['completed_mesh'] = face_substitution_path

        '''
        Face smooth
        '''
        if not (check_key(cfg, ['settings', 'face_smooth']) and not cfg.settings.face_smooth):
            # used inputs
            check_files(files, ['completed_mesh'])
            logging.info(f'Smooth face boundary.')

            smoothed_path = face_smooth(files['completed_mesh'], cfg_resources, uv_sampler, device, save_root=save_root)

            files['completed_mesh'] = smoothed_path


        '''
        ##############################
                Refinement
        ##############################
        '''

        '''
        Refinement
        '''
        if not (check_key(cfg, ['settings', 'refine']) and not cfg.settings.refine):
            # used inputs  (camera is not needed for ECON)
            check_files(files, ['completed_mesh', 'subsmplx', 'front_normal', 'back_normal', 'image', 'mask'])
            logging.info(f'Start \'Refinement\'.')

            # init model
            torch.cuda.empty_cache()
            cfg_refine = get_cfg_defaults(cfg_resources.configs.default_refine)
            cfg_regine = update_cfg_refine(cfg_refine, cfg)
            model = RefineUNet(cfg_refine.model)
            model.requires_grad_(False)
            model.eval()
            model.cuda()

            # load ckpt
            state_dict = torch.load(cfg_refine.test.ckpt_path, map_location=device)
            model.load_state_dict(state_dict['model'])
            logging.info(f'Load ckpt from {cfg_refine.test.ckpt_path} on {device}')

            # process
            test_dataset = RefineEvalDataset([files], cfg_refine.test, cfg_resources)
            evaluate_refine(uv_sampler, model, test_dataset, cfg_refine, cfg_resources, device, save_root=save_root)

        else:
            logging.info(f'Skip \'Refinement\'.')            

        '''
        ##############################
                 Modification
        ##############################
        '''

        '''
        Color projection (By default, we use the completed mesh)
        '''
        if not (check_key(cfg, ['settings', 'color_projection']) and not cfg.settings.color_projection):
            # used inputs  (camera is not needed for ECON)
            check_files(files, ['completed_mesh', 'image', 'mask'])

            if check_key(cfg, ['files', 'camera']):
                logging.info(f'Start \'Color projection\' with given camera parameters.')
                texture_projection(files, cfg, cfg_resources, uv_sampler, device, save_root=save_root, camera_param_path=files['camera'])
            else:
                logging.info(f'Start \'Color projection\' with default camera parameters.')
                texture_projection(files, cfg, cfg_resources, uv_sampler, device, save_root=save_root)

        # Use EMOCA facial texture
        # TODO: 

        # Down sample
        # TODO:

        # Animation
        # TODO:


    except Exception as e:
        logging.exception(e)