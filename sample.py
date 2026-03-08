import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from network import *
from models.GraphBrep import *
from diffusers import DDPMScheduler, PNDMScheduler,DPMSolverSDEScheduler
from OCC.Extend.DataExchange import write_stl_file, write_step_file
import random
import sde_lib
import yaml
from functools import partial
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sdesampling
from models.utils import score_fn
from utils import (
    randn_tensor,
    compute_bbox_center_and_size,
    generate_random_string,
    construct_brep,
    construct_brep_onlysurf,
    detect_shared_vertex,
    detect_shared_edge,
    detect_shared_edge2,
    joint_optimize,
    face_optimize,
    compute_boxx,
    edge_optimize,
    generate_mask,
    set_seed
)
import time
from datetime import datetime
import json

text2int = {'uncond':0, 
            'bathtub':1, 
            'bed':2, 
            'bench':3, 
            'bookshelf':4,
            'cabinet':5, 
            'chair':6, 
            'couch':7, 
            'lamp':8, 
            'sofa':9, 
            'table':10
            }
text2int_cadnet10 = {'uncond':0, 
                         'AngleIron': 1, 
                         'FlatKey': 2, 
                         'Grooved_Pin': 3, 
                         'HookWrench': 4, 
                         'Key': 5, 
                         'Spring': 6,
                        'Steel': 7, 'TangentialKey': 8, 'Thrust_Ring': 9, 'Washer': 10}

def sample(eval_args,iter,mode='furniture',result_status='result.txt'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = eval_args['batch_size']
    z_threshold = eval_args['z_threshold']
    bbox_threshold =eval_args['bbox_threshold']
    save_folder = eval_args['save_folder']
    num_surfaces = eval_args['num_surfaces'] 
    num_edges = eval_args['num_edges']

    if eval_args['use_cf']:
        if mode=='furniture':
            class_label = torch.LongTensor([text2int[eval_args['class_label']]]*batch_size + \
                                        [text2int['uncond']]*batch_size).cuda().reshape(-1,1) 
        else:
            class_label = torch.LongTensor([text2int_cadnet10[eval_args['class_label']]]*batch_size + \
                                        [text2int_cadnet10['uncond']]*batch_size).cuda().reshape(-1,1) 
        w = 0.6
    else:
        class_label = None

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    surfPos_model = SurfPosNet(eval_args['use_cf'])
    surfPos_model.load_state_dict(torch.load(eval_args['surfpos_weight'],weights_only=True))  
    surfPos_model = surfPos_model.to(device).eval()

    surfZ_model = SurfZNet(eval_args['use_cf'])
    surfZ_model.load_state_dict(torch.load(eval_args['surfz_weight'],weights_only=True))
    surfZ_model = surfZ_model.to(device).eval()


    edgePos_model = EdgePosNet(eval_args['use_cf'])
    edgePos_model.load_state_dict(torch.load(eval_args['edgepos_weight'],weights_only=True))
    edgePos_model = edgePos_model.to(device).eval()

    edgeZ_model = EdgeZNet(eval_args['use_cf'])
    edgeZ_model.load_state_dict(torch.load(eval_args['edgez_weight'],weights_only=True))
    edgeZ_model = edgeZ_model.to(device).eval()

    
    surf_vae = AutoencoderKLFastDecode(in_channels=3,
        out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512,
    )
    surf_vae.load_state_dict(torch.load(eval_args['surfvae_weight'],weights_only=True), strict=False)
    surf_vae = surf_vae.to(device).eval()

    edge_vae = AutoencoderKL1DFastDecode(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
        up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
        block_out_channels=[128, 256, 512],  
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512
    )
    edge_vae.load_state_dict(torch.load(eval_args['edgevae_weight'],weights_only=True), strict=False)
    edge_vae = edge_vae.to(device).eval()

    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
        clip_sample = True,
        clip_sample_range=3
    ) 
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(f'{save_folder}/samples/pkl'):
        os.makedirs(f'{save_folder}/samples/pkl',exist_ok=True)
    if not os.path.exists('{save_folder}/samples/xyzc'):
        os.makedirs(f'{save_folder}/samples/xyzc',exist_ok=True)
 

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            start_time = datetime.now()
            surfPos = randn_tensor((batch_size, num_surfaces, 6)).to(device)

            pndm_scheduler.set_timesteps(200)  
            for t in tqdm(pndm_scheduler.timesteps[:158]):#
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label)
                surfPos = pndm_scheduler.step(pred, t, surfPos).prev_sample
           
            if not eval_args['use_cf']:
                surfPos = surfPos.repeat(1,2,1)
                num_surfaces *= 2

            ddpm_scheduler.set_timesteps(1000)  
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):   
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label)
                surfPos = ddpm_scheduler.step(pred, t, surfPos).prev_sample
           

            surfPos_deduplicate = []
            surfMask_deduplicate = []
            for ii in range(batch_size):
                bboxes = np.round(surfPos[ii].unflatten(-1,torch.Size([2,3])).detach().cpu().numpy(), 4)   
                non_repeat = bboxes[:1]
                for bbox_idx, bbox in enumerate(bboxes):
                    diff = np.max(np.max(np.abs(non_repeat - bbox),-1),-1)
                    same = diff < bbox_threshold
                    bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                    diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev),-1),-1)
                    same_rev = diff_rev < bbox_threshold
                    if same.sum()>=1 or same_rev.sum()>=1:
                        continue # repeat value
                    else:
                        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]],0)
                bboxes = non_repeat.reshape(len(non_repeat),-1)

                surf_mask = torch.zeros((1, len(bboxes))) == 1
                bbox_padded = torch.concat([torch.FloatTensor(bboxes), torch.zeros(num_surfaces-len(bboxes),6)])
                mask_padded = torch.concat([surf_mask, torch.zeros(1, num_surfaces-len(bboxes))==0], -1)
                surfPos_deduplicate.append(bbox_padded)
                surfMask_deduplicate.append(mask_padded)

            surfPos = torch.stack(surfPos_deduplicate).cuda()
            surfMask = torch.vstack(surfMask_deduplicate).cuda()

            edgeM=surfMask.unsqueeze(-1) * surfMask.unsqueeze(1)
            face_mask=(~surfMask).to(torch.int)
            edge_mask=face_mask.unsqueeze(-1) * face_mask.unsqueeze(1)
            edge_mask = torch.tril(edge_mask, -1) + torch.tril(edge_mask, -1).transpose(-1, -2)
            edge_mask=edge_mask.unsqueeze(-1)

            surfZ = randn_tensor((batch_size, num_surfaces, 48)).to(device)
            
            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps): 
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    pred = surfZ_model(_surfZ_, timesteps, _surfPos_, _surfMask_, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfZ_model(surfZ, timesteps, surfPos, surfMask, class_label)
                surfZ = pndm_scheduler.step(pred, t, surfZ).prev_sample
            edgePos = randn_tensor((batch_size, num_surfaces, num_edges, 6)).cuda()
          
            pndm_scheduler.set_timesteps(200)  
            for t in tqdm(pndm_scheduler.timesteps[:158]):  
                timesteps = t.reshape(-1).cuda()   
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    _edgePos_ = edgePos.repeat(2,1,1,1)
                    noise_pred = edgePos_model(_edgePos_, timesteps, _surfPos_, _surfZ_, _surfMask_, class_label)
                    noise_pred = noise_pred[:batch_size] * (1+w) - noise_pred[batch_size:] * w
                else:
                    noise_pred = edgePos_model(edgePos, timesteps, surfPos, surfZ, surfMask, class_label)
                edgePos = pndm_scheduler.step(noise_pred, t, edgePos).prev_sample

            ddpm_scheduler.set_timesteps(1000)  
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):
                timesteps = t.reshape(-1).cuda()   
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    _edgePos_ = edgePos.repeat(2,1,1,1)
                    noise_pred = edgePos_model(_edgePos_, timesteps, _surfPos_, _surfZ_, _surfMask_, class_label)
                    noise_pred = noise_pred[:batch_size] * (1+w) - noise_pred[batch_size:] * w
                else:
                    noise_pred = edgePos_model(edgePos, timesteps, surfPos, surfZ, surfMask, class_label)
                edgePos = ddpm_scheduler.step(noise_pred, t, edgePos).prev_sample


            edgeM = surfMask.unsqueeze(-1).repeat(1, 1, num_edges)

            for ii in range(batch_size):
                edge_bboxs = edgePos[ii][~surfMask[ii]].detach().cpu().numpy()

                for surf_idx, bboxes in enumerate(edge_bboxs):
                    bboxes = bboxes.reshape(len(bboxes),2,3)
                    valid_bbox = bboxes[0:1]
                    for bbox_idx, bbox in enumerate(bboxes):
                        diff = np.max(np.max(np.abs(valid_bbox - bbox),-1),-1)
                        bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                        diff_rev = np.max(np.max(np.abs(valid_bbox - bbox_rev),-1),-1)
                        same = diff < bbox_threshold
                        same_rev = diff_rev < bbox_threshold
                        if same.sum()>=1 or same_rev.sum()>=1:
                            edgeM[ii, surf_idx, bbox_idx] = True
                            continue # repeat value
                        else:
                            valid_bbox = np.concatenate([valid_bbox, bbox[np.newaxis,:,:]],0)
                    edgeM[ii, surf_idx, 0] = False  # set first one to False  


            edgeZV = randn_tensor((batch_size, num_surfaces, num_edges, 18)).cuda()

            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps):
                timesteps = t.reshape(-1).cuda()   
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _edgePos_ = edgePos.repeat(2,1,1,1)
                    _edgeM_ = edgeM.repeat(2,1,1)
                    _edgeZV_ = edgeZV.repeat(2,1,1,1)
                    noise_pred = edgeZ_model(_edgeZV_, timesteps, _edgePos_, _surfPos_, _surfZ_, _edgeM_, class_label)
                    noise_pred = noise_pred[:batch_size] * (1+w) - noise_pred[batch_size:] * w
                else:
                    noise_pred = edgeZ_model(edgeZV, timesteps, edgePos, surfPos, surfZ, edgeM, class_label)
                edgeZV = pndm_scheduler.step(noise_pred, t, edgeZV).prev_sample

            edgeZV[edgeM] = 0 # set removed data to 0
            edge_z = edgeZV[:,:,:,:12]
            edgeV = edgeZV[:,:,:,12:].detach().cpu().numpy()

            surf_ncs = surf_vae(surfZ.unflatten(-1,torch.Size([16,3])).flatten(0,1).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
            surf_ncs = surf_ncs.permute(0,2,3,1).unflatten(0, torch.Size([batch_size, num_surfaces])).detach().cpu().numpy()
            
            edge_ncs = edge_vae(edge_z.unflatten(-1,torch.Size([4,3])).reshape(-1,4,3).permute(0,2,1))
            edge_ncs = edge_ncs.permute(0,2,1).reshape(batch_size, num_surfaces, num_edges, 32, 3).detach().cpu().numpy()
            end_time = datetime.now()
            ldm_time = (end_time - start_time)/batch_size
            edge_mask = edgeM.detach().cpu().numpy()     
            edge_pos = edgePos.detach().cpu().numpy() / 3.0
            surfPos = surfPos.detach().cpu().numpy()  / 3.0
            edge_ncs2= edgeV.reshape(batch_size, num_surfaces, num_edges, 2, 3)
    face_mask_tem = face_mask.bool().detach().cpu().numpy()
    edge_mask_tem = (~edgeM).detach().cpu().numpy()
    for batch_idx in range(batch_size):
        pkl_path=save_pkl(surf_ncs[batch_idx][face_mask_tem[batch_idx]],surfPos[batch_idx][face_mask_tem[batch_idx]],
                            edge_ncs[batch_idx],edge_pos[batch_idx],
                            edge_ncs2[batch_idx],
                            edge_mask_tem[batch_idx],
                            f'{save_folder}/samples/pkl/test_{iter}_{batch_idx}.pkl')
        surf_ncs_try = face_optimize(surf_ncs[batch_idx][face_mask_tem[batch_idx]],surfPos[batch_idx][face_mask_tem[batch_idx]],f'{save_folder}/samples/xyzc/test_face_{batch_idx}.xyzc')
        
        _ = edge_optimize(edge_ncs[batch_idx],edge_pos[batch_idx],
                                 edge_mask_tem[batch_idx],
                                 f'{save_folder}/samples/xyzc/test_edge_{iter}_{batch_idx}.xyzc',focu=True)   
        _ = edge_optimize(edge_ncs2[batch_idx],edge_pos[batch_idx],
                                  edge_mask_tem[batch_idx],
                                  f'{save_folder}/samples/xyzc/test_v_{iter}_{batch_idx}.xyzc',focu=False)  
    
    succ=0
    for batch_idx in range(batch_size):
        start_time=datetime.now()
        surfMask_cad = surfMask[batch_idx].detach().cpu().numpy()
        edge_mask_cad = edge_mask[batch_idx][~surfMask_cad]
        edge_pos_cad = edge_pos[batch_idx][~surfMask_cad]
        edge_ncs_cad = edge_ncs[batch_idx][~surfMask_cad]
        edgeV_cad = edgeV[batch_idx][~surfMask_cad]
        edge_z_cad = edge_z[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()[~edge_mask_cad]
        surf_z_cad = surfZ[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()
        surf_pos_cad = surfPos[batch_idx][~surfMask_cad]

        edgeV_bbox = []
        for bbox, ncs, mask in zip(edge_pos_cad, edge_ncs_cad, edge_mask_cad):
            epos = bbox[~mask]
            edge = ncs[~mask]
            bbox_startends = []
            for bb, ee in zip(epos, edge): 
                bcenter, bsize = compute_bbox_center_and_size(bb[0:3], bb[3:])
                wcs = ee*(bsize/2) + bcenter
                bbox_start_end = wcs[[0,-1]]
                bbox_start_end = bbox_start_end.reshape(2,3)
                bbox_startends.append(bbox_start_end.reshape(1,2,3))
            bbox_startends = np.vstack(bbox_startends)
            edgeV_bbox.append(bbox_startends)
        
        try:
            unique_vertices, new_vertex_dict = detect_shared_vertex(edgeV_cad, edge_mask_cad, edgeV_bbox)
        except Exception as e:
            end_time = datetime.now()
            post_time = end_time - start_time
            save_sampling_result(eval_args, iter, batch_idx, 'ver-faild', ldm_time.total_seconds(),post_time.total_seconds(), result_status)
            print('Vertex detection failed...')
            continue
        
        try:
            unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj = detect_shared_edge(unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, z_threshold, edge_mask_cad)
        except Exception as e:
            end_time = datetime.now()
            post_time = end_time - start_time
            save_sampling_result(eval_args, iter, batch_idx, 'edge-faild', ldm_time.total_seconds(),post_time.total_seconds(), result_status)
            print('Edge detection failed...')
            continue
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                surf_ncs_cad = surf_vae(torch.FloatTensor(unique_faces).cuda().unflatten(-1,torch.Size([16,3])).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
                surf_ncs_cad = surf_ncs_cad.permute(0,2,3,1).detach().cpu().numpy()
                edge_ncs_cad = edge_vae(torch.FloatTensor(unique_edges).cuda().unflatten(-1,torch.Size([4,3])).permute(0,2,1))
                edge_ncs_cad = edge_ncs_cad.permute(0,2,1).detach().cpu().numpy()

        num_edge = len(edge_ncs_cad)
        num_surf = len(surf_ncs_cad)
        surf_wcs, edge_wcs = joint_optimize(surf_ncs_cad, edge_ncs_cad, surf_pos_cad, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf)
        
        try:
            solid = construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj)
        except Exception as e:
            end_time = datetime.now()
            post_time = end_time - start_time

            save_sampling_result(eval_args, iter, batch_idx, 'build-faild', ldm_time.total_seconds(),post_time.total_seconds(), result_status)
            print('B-rep rebuild failed...')
            continue
        end_time = datetime.now()
        post_time = end_time - start_time
        random_string = generate_random_string(15)
        write_step_file(solid, f'{save_folder}/{iter}_{batch_idx}.step')
        write_stl_file(solid, f'{save_folder}/{iter}_{batch_idx}.stl', linear_deflection=0.001, angular_deflection=0.5)
        save_sampling_result(eval_args, iter, batch_idx, 'success',ldm_time.total_seconds(), post_time.total_seconds(), result_status)
        succ+=1
    return succ




def sample_GBrep(eval_args,iter,result_status='result.txt'):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = eval_args['batch_size']
    z_threshold = eval_args['z_threshold']
    bbox_threshold =eval_args['bbox_threshold']
    save_folder = eval_args['save_folder']
    num_surfaces = eval_args['num_surfaces'] 
    num_edges = eval_args['num_edges']

    if eval_args['use_cf']:
        class_label = torch.LongTensor([text2int[eval_args['class_label']]]*batch_size + \
                                       [text2int['uncond']]*batch_size).cuda().reshape(-1,1) 
        w = 0.6
    else:
        class_label = None
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    surfPos_model = SurfPosNet(eval_args['use_cf'])
    surfPos_model.load_state_dict(torch.load(eval_args['surfpos_weight']))  
    surfPos_model = surfPos_model.to(device).eval()

    surfZ_model = SurfZNet(eval_args['use_cf'])
    surfZ_model.load_state_dict(torch.load(eval_args['surfz_weight']))
    surfZ_model = surfZ_model.to(device).eval()


    edgePos_model = EdgePosNet_refine(eval_args['use_cf'])
    edgePos_model.load_state_dict(torch.load(eval_args['GEdgePos_weight']))
    edgePos_model = edgePos_model.to(device).eval()

    edgeZ_model = EdgeZNet_refine(eval_args['use_cf'])
    edgeZ_model.load_state_dict(torch.load(eval_args['GEdgeZ_weight']))
    edgeZ_model = edgeZ_model.to(device).eval()

    
    surf_vae = AutoencoderKLFastDecode(in_channels=3,
        out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512,
    )
    surf_vae.load_state_dict(torch.load(eval_args['surfvae_weight']), strict=False)
    surf_vae = surf_vae.to(device).eval()

    edge_vae = AutoencoderKL1DFastDecode(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
        up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
        block_out_channels=[128, 256, 512],  
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512
    )
    edge_vae.load_state_dict(torch.load(eval_args['edgevae_weight']), strict=False)
    edge_vae = edge_vae.to(device).eval()

    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
        clip_sample = True,
        clip_sample_range=3
    ) 
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(f'{save_folder}/samples/{iter}'):
        os.makedirs(f'{save_folder}/samples/{iter}')
 
    torch.cuda.empty_cache()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            start_time = datetime.now()
            surfPos = randn_tensor((batch_size, num_surfaces, 6)).to(device)

            pndm_scheduler.set_timesteps(200)  
            for t in tqdm(pndm_scheduler.timesteps[:158]):#
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label)
                surfPos = pndm_scheduler.step(pred, t, surfPos).prev_sample
           
            if not eval_args['use_cf']:
                surfPos = surfPos.repeat(1,2,1)
                num_surfaces *= 2

            ddpm_scheduler.set_timesteps(1000)  
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):   
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label)
                surfPos = ddpm_scheduler.step(pred, t, surfPos).prev_sample
           

            surfPos_deduplicate = []
            surfMask_deduplicate = []
            for ii in range(batch_size):
                bboxes = np.round(surfPos[ii].unflatten(-1,torch.Size([2,3])).detach().cpu().numpy(), 4)   
                non_repeat = bboxes[:1]
                for bbox_idx, bbox in enumerate(bboxes):
                    diff = np.max(np.max(np.abs(non_repeat - bbox),-1),-1)
                    same = diff < bbox_threshold
                    bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                    diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev),-1),-1)
                    same_rev = diff_rev < bbox_threshold
                    if same.sum()>=1 or same_rev.sum()>=1:
                        continue # repeat value
                    else:
                        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]],0)
                bboxes = non_repeat.reshape(len(non_repeat),-1)

                surf_mask = torch.zeros((1, len(bboxes))) == 1
                bbox_padded = torch.concat([torch.FloatTensor(bboxes), torch.zeros(num_surfaces-len(bboxes),6)])
                mask_padded = torch.concat([surf_mask, torch.zeros(1, num_surfaces-len(bboxes))==0], -1)
                surfPos_deduplicate.append(bbox_padded)
                surfMask_deduplicate.append(mask_padded)
            torch.cuda.empty_cache()
            surfPos = torch.stack(surfPos_deduplicate).cuda()
            surfMask = torch.vstack(surfMask_deduplicate).cuda()
            surfPos=surfPos[:,:num_edges,:]
            surfMask=surfMask[:,:num_edges]

            edgeM=surfMask.unsqueeze(-1) * surfMask.unsqueeze(1)
            face_mask=(~surfMask).to(torch.int)
            edge_mask=face_mask.unsqueeze(-1) * face_mask.unsqueeze(1)
            edge_mask = torch.tril(edge_mask, -1) + torch.tril(edge_mask, -1).transpose(-1, -2)
            edge_mask=edge_mask.unsqueeze(-1)

            surfZ = randn_tensor((batch_size, num_edges, 48)).to(device)
            
            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps): 
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    pred = surfZ_model(_surfZ_, timesteps, _surfPos_, _surfMask_, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfZ_model(surfZ, timesteps, surfPos, surfMask, class_label)
                surfZ = pndm_scheduler.step(pred, t, surfZ).prev_sample
            torch.cuda.empty_cache()
            surfPos=surfPos / 3.0
            edgePos= randn_tensor((batch_size,6,num_edges,num_edges)).cuda()
            edgePos=(torch.tril(edgePos, -1) + torch.tril(edgePos, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  
            pndm_scheduler.set_timesteps(200)  
            for t in tqdm(pndm_scheduler.timesteps[:158]):
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _face_mask_ = face_mask.repeat(2,1)
                    _edgePos_ = edgePos.repeat(2,1,1,1)
                    _edge_mask_ = edge_mask.repeat(2,1,1,1)
                    noise_pred = edgePos_model(_edgePos_,_surfPos_, _surfZ_,timesteps,_face_mask_,_edge_mask_,class_label)
                    noise_pred = noise_pred[:batch_size] * (1+w) - noise_pred[batch_size:] * w
                    noise_pred = noise_pred.permute(0, 3, 1, 2)
                    noise_pred=(torch.tril(noise_pred, -1) + torch.tril(noise_pred, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask 
                else:
                    noise_pred = edgePos_model(edgePos, surfPos, surfZ,timesteps,face_mask,edge_mask, class_label)
                edgePos = pndm_scheduler.step(noise_pred, t, edgePos).prev_sample
                edgePos = edgePos.permute(0, 3, 1, 2)
                edgePos=(torch.tril(edgePos, -1) + torch.tril(edgePos, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  

            ddpm_scheduler.set_timesteps(1000)  
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):
                timesteps = t.reshape(-1).cuda()  
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _face_mask_ = face_mask.repeat(2,1)
                    _edgePos_ = edgePos.repeat(2,1,1,1)
                    _edge_mask_ = edge_mask.repeat(2,1,1,1)

                    noise_pred = edgePos_model(_edgePos_,_surfPos_, _surfZ_,timesteps,_face_mask_,_edge_mask_,class_label)
                    noise_pred = noise_pred[:batch_size] * (1+w) - noise_pred[batch_size:] * w
                    noise_pred = noise_pred.permute(0, 3, 1, 2)
                    noise_pred=(torch.tril(noise_pred, -1) + torch.tril(noise_pred, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask 
                else:
                    noise_pred = edgePos_model(edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
                edgePos = ddpm_scheduler.step(noise_pred, t, edgePos).prev_sample
                edgePos = edgePos.permute(0, 3, 1, 2)
                edgePos=(torch.tril(edgePos, -1) + torch.tril(edgePos, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  
            
            torch.cuda.empty_cache()
            edgeM=((edgePos<0.2).all(-1) & (edgePos>-0.2).all(-1))
            edge_mask=(~edgeM).float().unsqueeze(-1)
            edgeZV= randn_tensor((batch_size,18,num_edges,num_edges)).cuda()
            edgeZV=(torch.tril(edgeZV, -1) + torch.tril(edgeZV, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  
            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps):
                timesteps = t.reshape(-1).cuda()   
                noise_pred = edgeZ_model(edgeZV,edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
                edgeZV = pndm_scheduler.step(noise_pred, t, edgeZV).prev_sample
                edgeZV = edgeZV.permute(0, 3, 1, 2)
                edgeZV=(torch.tril(edgeZV, -1) + torch.tril(edgeZV, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask 

            edgePos=edgePos*3
            surfPos=surfPos*3
           
            
            edgeZV=(edgeZV+edgeZV.transpose(1, 2))/2
            edge_z = edgeZV[:,:,:,:12]
            edgeV = edgeZV[:,:,:,12:].detach()
            

            edgePos[edgeM] = 0 # set removed data to 0
            edge_z[edgeM] = 0 # set removed data to 0
            edgeV[edgeM] = 0 # set removed data to 0
            edgeV = edgeV.cpu().numpy()
            
            
            surf_ncs = surf_vae(surfZ.unflatten(-1,torch.Size([16,3])).flatten(0,1).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
            surf_ncs = surf_ncs.permute(0,2,3,1).unflatten(0, torch.Size([batch_size, num_edges])).detach().cpu().numpy()
            
            edge_ncs = edge_vae(edge_z.unflatten(-1,torch.Size([4,3])).reshape(-1,4,3).permute(0,2,1))
            edge_ncs = edge_ncs.permute(0,2,1).reshape(batch_size, num_edges, num_edges, 32, 3).detach().cpu().numpy()
            surfMask=edgeM.all(dim=-1)
            edge_mask = edgeM.detach().cpu().numpy()     
            edge_pos = edgePos.detach().cpu().numpy() / 3.0
            surfPos = surfPos.detach().cpu().numpy()  / 3.0
            edge_ncs2= edgeV.reshape(batch_size, num_edges, num_edges, 2, 3)
    torch.cuda.empty_cache()
    face_mask_tem = face_mask.bool().detach().cpu().numpy()
    edge_mask_tem = (~edgeM).detach().cpu().numpy()
    for batch_idx in range(batch_size):
        surf_wcs = face_optimize(surf_ncs[batch_idx][face_mask_tem[batch_idx]],surfPos[batch_idx][face_mask_tem[batch_idx]],f'{save_folder}/samples/{iter}/test_face_{batch_idx}.xyzc')
        edge_wcs = edge_optimize(edge_ncs[batch_idx],edge_pos[batch_idx],
                                 edge_mask_tem[batch_idx],
                                 f'{save_folder}/samples/{iter}/test_edge_{batch_idx}.xyzc',focu=True)   
        edge_wcs2 = edge_optimize(edge_ncs2[batch_idx],edge_pos[batch_idx],
                                  edge_mask_tem[batch_idx],
                                  f'{save_folder}/samples/{iter}/test_edge_v_{batch_idx}.xyzc',focu=False)   
    for batch_idx in range(batch_size):
        surfMask_cad = surfMask[batch_idx].detach().cpu().numpy()
        edge_mask_cad = edge_mask[batch_idx][~surfMask_cad]
        edge_pos_cad = edge_pos[batch_idx][~surfMask_cad]
        edge_ncs_cad = edge_ncs[batch_idx][~surfMask_cad]
        edgeV_cad = edgeV[batch_idx][~surfMask_cad]
        edge_z_cad = edge_z[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()[~edge_mask_cad]
        surf_z_cad = surfZ[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()
        surf_pos_cad = surfPos[batch_idx][~surfMask_cad]

        edgeV_bbox = []
        for bbox, ncs, mask in zip(edge_pos_cad, edge_ncs_cad, edge_mask_cad):
            epos = bbox[~mask]
            edge = ncs[~mask]
            bbox_startends = []
            for bb, ee in zip(epos, edge): 
                bcenter, bsize = compute_bbox_center_and_size(bb[0:3], bb[3:])
                wcs = ee*(bsize/2) + bcenter
                bbox_start_end = wcs[[0,-1]]
                bbox_start_end = bbox_start_end.reshape(2,3)
                bbox_startends.append(bbox_start_end.reshape(1,2,3))
            bbox_startends = np.vstack(bbox_startends)
            edgeV_bbox.append(bbox_startends)
        
        try:
            unique_vertices, new_vertex_dict = detect_shared_vertex(edgeV_cad, edge_mask_cad, edgeV_bbox)
        except Exception as e:
            print('Vertex detection failed...')
            continue
        
        try:
            unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj = detect_shared_edge2(unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, edge_mask_cad)
        except Exception as e:
            print('Edge detection failed...')
            continue
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                surf_ncs_cad = surf_vae(torch.FloatTensor(unique_faces).cuda().unflatten(-1,torch.Size([16,3])).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
                surf_ncs_cad = surf_ncs_cad.permute(0,2,3,1).detach().cpu().numpy()
                edge_ncs_cad = edge_vae(torch.FloatTensor(unique_edges).cuda().unflatten(-1,torch.Size([4,3])).permute(0,2,1))
                edge_ncs_cad = edge_ncs_cad.permute(0,2,1).detach().cpu().numpy()

        num_edge = len(edge_ncs_cad)
        num_surf = len(surf_ncs_cad)
        surf_wcs, edge_wcs = joint_optimize(surf_ncs_cad, edge_ncs_cad, surf_pos_cad, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf)
        
        try:
            solid = construct_brep(surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj)
        except Exception as e:
            print('B-rep rebuild failed...')
            continue

        random_string = generate_random_string(15)
        write_step_file(solid, f'{save_folder}/{iter}_{batch_idx}.step')
        write_stl_file(solid, f'{save_folder}/{iter}_{batch_idx}.stl', linear_deflection=0.001, angular_deflection=0.5)

        if False:    
            evp = evp* edge_mask # set removed data to 0
            edgePos=evp[:,:,:,:6]
            edgeV=evp[:,:,:,6:]

            surf_ncs = surf_vae(surfZ.unflatten(-1,torch.Size([16,3])).flatten(0,1).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
            surf_ncs = surf_ncs.permute(0,2,3,1).unflatten(0, torch.Size([batch_size, num_surfaces])).detach().cpu().numpy()
            
            edge_ncs = edgePos.reshape(batch_size, num_surfaces, num_surfaces, 2, 3).detach().cpu().numpy()
            edge_ncs2= edgeV.reshape(batch_size, num_surfaces, num_surfaces, 2, 3).detach().cpu().numpy()
            adj_evp=(~((evp<0.5).all(-1) & (evp>-0.5).all(-1)))
            
            edge_mask = edgeM.detach().cpu().numpy()     
            edgeV = edgeV.detach().cpu().numpy() / 3.0
            edge_pos = edgePos.detach().cpu().numpy() / 3.0
            surfPos = surfPos.detach().cpu().numpy()  / 3.0
            face_mask = face_mask.bool().detach().cpu().numpy()
            edge_mask = adj_evp.squeeze(-1).bool().detach().cpu().numpy()
            for batch_idx in range(batch_size):
                surf_wcs = face_optimize(surf_ncs[batch_idx][face_mask[batch_idx]],surfPos[batch_idx][face_mask[batch_idx]],f'{save_folder}/samples/{iter}/test_face_{batch_idx}.xyzc')
                edge_wcs = edge_optimize(edge_ncs[batch_idx][edge_mask[batch_idx]],edge_pos[batch_idx][edge_mask[batch_idx]],f'{save_folder}/samples/{iter}/test_edge_{batch_idx}.xyzc',focu=False)   
                edge_wcs2 = edge_optimize(edge_ncs2[batch_idx][edge_mask[batch_idx]],edgeV[batch_idx][edge_mask[batch_idx]],f'{save_folder}/samples/{iter}/test_edge_v_{batch_idx}.xyzc',focu=False)   


def sample_GBrep_noC(eval_args,iter,mode='furniture',result_status='result.txt'):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = eval_args['batch_size']
    z_threshold = eval_args['z_threshold']
    bbox_threshold =eval_args['bbox_threshold']
    save_folder = eval_args['save_folder']
    num_surfaces = eval_args['num_surfaces'] 
    num_edges = eval_args['num_edges']

    if eval_args['use_cf']:
        if mode=='furniture':
            class_label = torch.LongTensor([text2int[eval_args['class_label']]]*batch_size + \
                                        [text2int['uncond']]*batch_size).cuda().reshape(-1,1) 
        else:
            class_label = torch.LongTensor([text2int_cadnet10[eval_args['class_label']]]*batch_size + \
                                        [text2int_cadnet10['uncond']]*batch_size).cuda().reshape(-1,1) 
        w = 0.6
    else:
        class_label = None
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    

    surfPos_model = SurfPosNet(eval_args['use_cf'])
    surfPos_model.load_state_dict(torch.load(eval_args['surfpos_weight'],weights_only=True))  
    surfPos_model = surfPos_model.to(device).eval()

    surfZ_model = SurfZNet(eval_args['use_cf'])
    surfZ_model.load_state_dict(torch.load(eval_args['surfz_weight'],weights_only=True))
    surfZ_model = surfZ_model.to(device).eval()


    edgeZ_model = CGTD_EZ(eval_args['use_cf'],max_p=False)
    edgePos_model = CGTD_EP(eval_args['use_cf'])



    
    edgePos_model.load_state_dict(torch.load(eval_args['GEdgePos_weight']))
    edgePos_model = edgePos_model.to(device).eval()

    edgeZ_model.load_state_dict(torch.load(eval_args['GEdgeZ_weight'],weights_only=True))
    edgeZ_model = edgeZ_model.to(device).eval()

    
    surf_vae = AutoencoderKLFastDecode(in_channels=3,
        out_channels=3,
        down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types= ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels=[128, 256, 512, 512],
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512,
    )
    surf_vae.load_state_dict(torch.load(eval_args['surfvae_weight'],weights_only=True), strict=False)
    surf_vae = surf_vae.to(device).eval()

    edge_vae = AutoencoderKL1DFastDecode(
        in_channels=3,
        out_channels=3,
        down_block_types=['DownBlock1D', 'DownBlock1D', 'DownBlock1D'],
        up_block_types=['UpBlock1D', 'UpBlock1D', 'UpBlock1D'],
        block_out_channels=[128, 256, 512],  
        layers_per_block=2,
        act_fn='silu',
        latent_channels=3,
        norm_num_groups=32,
        sample_size=512
    )
    edge_vae.load_state_dict(torch.load(eval_args['edgevae_weight'],weights_only=True), strict=False)
    edge_vae = edge_vae.to(device).eval()

    pndm_scheduler = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
    )

    ddpm_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
        clip_sample = True,
        clip_sample_range=3
    ) 



    pndm_scheduler_e = PNDMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
    )

    ddpm_scheduler_e = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='linear',
        prediction_type='epsilon',
        beta_start = 0.0001,
        beta_end = 0.02,
        clip_sample = True,
        clip_sample_range=3
    ) 
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(f'{save_folder}/samples/pkl'):
        os.makedirs(f'{save_folder}/samples/pkl',exist_ok=True)
    if not os.path.exists('{save_folder}/samples/xyzc'):
        os.makedirs(f'{save_folder}/samples/xyzc',exist_ok=True)
    succ=0
    edge_threshold = eval_args['edge_threshold']
    print(f"Edge threshold set to: {edge_threshold}")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            start_time = datetime.now()
            surfPos = randn_tensor((batch_size, num_surfaces, 6)).to(device)

            pndm_scheduler.set_timesteps(200)  
            for t in tqdm(pndm_scheduler.timesteps[:158]):#
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label)
                surfPos = pndm_scheduler.step(pred, t, surfPos).prev_sample
           
            if not eval_args['use_cf']:
                surfPos = surfPos.repeat(1,2,1)
                num_surfaces *= 2

            ddpm_scheduler.set_timesteps(1000)  
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):   
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfPos_ = surfPos.repeat(2,1,1)
                    pred = surfPos_model(_surfPos_, timesteps, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfPos_model(surfPos, timesteps, class_label)
                surfPos = ddpm_scheduler.step(pred, t, surfPos).prev_sample
           

            surfPos_deduplicate = []
            surfMask_deduplicate = []
            for ii in range(batch_size):
                bboxes = np.round(surfPos[ii].unflatten(-1,torch.Size([2,3])).detach().cpu().numpy(), 4)   
                non_repeat = bboxes[:1]
                for bbox_idx, bbox in enumerate(bboxes):
                    diff = np.max(np.max(np.abs(non_repeat - bbox),-1),-1)
                    same = diff < bbox_threshold
                    bbox_rev = bbox[::-1]  # also test reverse bbox for matching
                    diff_rev = np.max(np.max(np.abs(non_repeat - bbox_rev),-1),-1)
                    same_rev = diff_rev < bbox_threshold
                    if same.sum()>=1 or same_rev.sum()>=1:
                        continue # repeat value
                    else:
                        non_repeat = np.concatenate([non_repeat, bbox[np.newaxis,:,:]],0)
                bboxes = non_repeat.reshape(len(non_repeat),-1)

                surf_mask = torch.zeros((1, len(bboxes))) == 1
                bbox_padded = torch.concat([torch.FloatTensor(bboxes), torch.zeros(num_surfaces-len(bboxes),6)])
                mask_padded = torch.concat([surf_mask, torch.zeros(1, num_surfaces-len(bboxes))==0], -1)
                surfPos_deduplicate.append(bbox_padded)
                surfMask_deduplicate.append(mask_padded)
            surfPos = torch.stack(surfPos_deduplicate).cuda()
            surfMask = torch.vstack(surfMask_deduplicate).cuda()
            surfPos=surfPos[:,:num_edges,:]
            surfMask=surfMask[:,:num_edges]

            edgeM=surfMask.unsqueeze(-1) * surfMask.unsqueeze(1)
            face_mask=(~surfMask).to(torch.int)
            edge_mask=face_mask.unsqueeze(-1) * face_mask.unsqueeze(1)
            edge_mask = torch.tril(edge_mask, -1) + torch.tril(edge_mask, -1).transpose(-1, -2)
            edge_mask=edge_mask.unsqueeze(-1)

            surfZ = randn_tensor((batch_size, num_edges, 48)).to(device)
            
            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps): 
                timesteps = t.reshape(-1).cuda()
                if class_label is not None:
                    _surfZ_ = surfZ.repeat(2,1,1)
                    _surfPos_ = surfPos.repeat(2,1,1)
                    _surfMask_ = surfMask.repeat(2,1)
                    pred = surfZ_model(_surfZ_, timesteps, _surfPos_, _surfMask_, class_label)
                    pred = pred[:batch_size] * (1+w) - pred[batch_size:] * w
                else:
                    pred = surfZ_model(surfZ, timesteps, surfPos, surfMask, class_label)
                surfZ = pndm_scheduler.step(pred, t, surfZ).prev_sample
            surfPos=surfPos / 3.0
            edgePos= randn_tensor((batch_size,6,num_edges,num_edges)).cuda()
            edgePos=(torch.tril(edgePos, -1) + torch.tril(edgePos, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  
            pndm_scheduler_e.set_timesteps(200)  
            for t in tqdm(pndm_scheduler_e.timesteps[:158]):
                timesteps = t.reshape(-1).cuda()
                timesteps=timesteps.repeat(batch_size)
                noise_pred = edgePos_model(edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
                edgePos = pndm_scheduler_e.step(noise_pred, t, edgePos).prev_sample
                edgePos = edgePos.permute(0, 3, 1, 2)
                edgePos=(torch.tril(edgePos, -1) + torch.tril(edgePos, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  

            ddpm_scheduler_e.set_timesteps(1000)  
            for t in tqdm(ddpm_scheduler_e.timesteps[-250:]):
                timesteps = t.reshape(-1).cuda()  

                timesteps=timesteps.repeat(batch_size)
                noise_pred = edgePos_model(edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
                edgePos = ddpm_scheduler_e.step(noise_pred, t, edgePos).prev_sample
                edgePos = edgePos.permute(0, 3, 1, 2)
                edgePos=(torch.tril(edgePos, -1) + torch.tril(edgePos, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  
            

            edgeM=((edgePos<edge_threshold).all(-1) & (edgePos>-edge_threshold).all(-1))
            edge_mask=(~edgeM).float().unsqueeze(-1)
            edgeZV= randn_tensor((batch_size,18,num_edges,num_edges)).cuda()
            edgeZV=(torch.tril(edgeZV, -1) + torch.tril(edgeZV, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  
            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps):
                timesteps = t.reshape(-1).cuda()   
                noise_pred = edgeZ_model(edgeZV,edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
                timesteps=timesteps.repeat(batch_size)
                edgeZV = pndm_scheduler.step(noise_pred, t, edgeZV).prev_sample
                edgeZV = edgeZV.permute(0, 3, 1, 2)
                edgeZV=(torch.tril(edgeZV, -1) + torch.tril(edgeZV, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask 

            edgePos=edgePos*3
            surfPos=surfPos*3
           
            
            edgeZV=(edgeZV+edgeZV.transpose(1, 2))/2
            edge_z = edgeZV[:,:,:,:12]
            edgeV = edgeZV[:,:,:,12:].detach()
            

            edgePos[edgeM] = 0 # set removed data to 0
            edge_z[edgeM] = 0 # set removed data to 0
            edgeV[edgeM] = 0 # set removed data to 0
            edgeV = edgeV.cpu().numpy()
            
            
            surf_ncs = surf_vae(surfZ.unflatten(-1,torch.Size([16,3])).flatten(0,1).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
            surf_ncs = surf_ncs.permute(0,2,3,1).unflatten(0, torch.Size([batch_size, num_edges])).detach().cpu().numpy()
            
            edge_ncs = edge_vae(edge_z.unflatten(-1,torch.Size([4,3])).reshape(-1,4,3).permute(0,2,1))
            edge_ncs = edge_ncs.permute(0,2,1).reshape(batch_size, num_edges, num_edges, 32, 3).detach().cpu().numpy()
            end_time = datetime.now()
            ldm_time = (end_time - start_time)/batch_size

            surfMask=edgeM.all(dim=-1)
            edge_mask = edgeM.detach().cpu().numpy()     
            edge_pos = edgePos.detach().cpu().numpy() / 3.0
            surfPos = surfPos.detach().cpu().numpy()  / 3.0
            edge_ncs2= edgeV.reshape(batch_size, num_edges, num_edges, 2, 3)
    face_mask_tem = face_mask.bool().detach().cpu().numpy()
    edge_mask_tem = (~edgeM).detach().cpu().numpy()
    for batch_idx in range(batch_size):
        pkl_path=save_pkl(surf_ncs[batch_idx][face_mask_tem[batch_idx]],surfPos[batch_idx][face_mask_tem[batch_idx]],
                            edge_ncs[batch_idx],edge_pos[batch_idx],
                            edge_ncs2[batch_idx],
                            edge_mask_tem[batch_idx],
                            f'{save_folder}/samples/pkl/test_{iter}_{batch_idx}.pkl')
        surf_ncs_try = face_optimize(surf_ncs[batch_idx][face_mask_tem[batch_idx]],surfPos[batch_idx][face_mask_tem[batch_idx]],f'{save_folder}/samples/xyzc/test_face_{iter}_{batch_idx}.xyzc')
        
        _ = edge_optimize(edge_ncs[batch_idx],edge_pos[batch_idx],
                                 edge_mask_tem[batch_idx],
                                 f'{save_folder}/samples/xyzc/test_edge_{iter}_{batch_idx}.xyzc',focu=True)   
        _ = edge_optimize(edge_ncs2[batch_idx],edge_pos[batch_idx],
                                  edge_mask_tem[batch_idx],
                                  f'{save_folder}/samples/xyzc/test_v_{iter}_{batch_idx}.xyzc',focu=False) 
    for batch_idx in range(batch_size):
        start_time = datetime.now()
        surfMask_cad = surfMask[batch_idx].detach().cpu().numpy()
        edge_mask_cad = edge_mask[batch_idx][~surfMask_cad]
        edge_pos_cad = edge_pos[batch_idx][~surfMask_cad]
        edge_ncs_cad = edge_ncs[batch_idx][~surfMask_cad]
        edgeV_cad = edgeV[batch_idx][~surfMask_cad]
        edge_z_cad = edge_z[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()[~edge_mask_cad]
        surf_z_cad = surfZ[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()
        surf_pos_cad = surfPos[batch_idx][~surfMask_cad]

        edgeV_bbox = []
        for bbox, ncs, mask in zip(edge_pos_cad, edge_ncs_cad, edge_mask_cad):
            epos = bbox[~mask]
            edge = ncs[~mask]
            bbox_startends = []
            for bb, ee in zip(epos, edge): 
                bcenter, bsize = compute_bbox_center_and_size(bb[0:3], bb[3:])
                wcs = ee*(bsize/2) + bcenter
                bbox_start_end = wcs[[0,-1]]
                bbox_start_end = bbox_start_end.reshape(2,3)
                bbox_startends.append(bbox_start_end.reshape(1,2,3))
            bbox_startends = np.vstack(bbox_startends)
            edgeV_bbox.append(bbox_startends)
        
        try:
            unique_vertices, new_vertex_dict = detect_shared_vertex2(edgeV_cad, edge_mask_cad, edgeV_bbox)
        except Exception as e:
            end_time = datetime.now()
            post_time = end_time - start_time
            save_sampling_result(eval_args, iter, batch_idx, 'ver-faild', ldm_time.total_seconds(),post_time.total_seconds(), result_status)
            print('Vertex detection failed...')
            continue
        
        try:
            unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj = detect_shared_edge2(unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, edge_mask_cad)
        except Exception as e:
            end_time = datetime.now()
            post_time = end_time - start_time
            save_sampling_result(eval_args, iter, batch_idx, 'edge-faild', ldm_time.total_seconds(),post_time.total_seconds(), result_status)
            print('Edge detection failed...')
            continue
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                surf_ncs_cad = surf_vae(torch.FloatTensor(unique_faces).cuda().unflatten(-1,torch.Size([16,3])).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
                surf_ncs_cad = surf_ncs_cad.permute(0,2,3,1).detach().cpu().numpy()
                edge_ncs_cad = edge_vae(torch.FloatTensor(unique_edges).cuda().unflatten(-1,torch.Size([4,3])).permute(0,2,1))
                edge_ncs_cad = edge_ncs_cad.permute(0,2,1).detach().cpu().numpy()

        num_edge = len(edge_ncs_cad)
        num_surf = len(surf_ncs_cad)
        surf_wcs, edge_wcs = joint_optimize(surf_ncs_cad, edge_ncs_cad, surf_pos_cad, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf)
        
        with ThreadPoolExecutor() as executor:
            future = executor.submit(construct_brep, surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj)
            try:
                # 设置任务超时时间为 5 秒
                solid = future.result(timeout=10)
                write_step_file(solid, f'{save_folder}/{iter}_{batch_idx}.step')
                write_stl_file(solid, f'{save_folder}/{iter}_{batch_idx}.stl', linear_deflection=0.001, angular_deflection=0.5)
                end_time = datetime.now()
                post_time = end_time - start_time
                save_sampling_result(eval_args, iter, batch_idx, 'success', ldm_time.total_seconds(),post_time.total_seconds(), result_status)
                succ+=1
            
            except TimeoutError:
                # 捕获超时异常
                end_time = datetime.now()
                post_time = end_time - start_time
                save_sampling_result(eval_args, iter, batch_idx, 'build-timeout', ldm_time.total_seconds(), post_time.total_seconds(), result_status)
                print("B-rep rebuild timed out.")
                continue
            
            except Exception as e:
                # 捕获其他异常
                end_time = datetime.now()
                post_time = end_time - start_time
                save_sampling_result(eval_args, iter, batch_idx, 'build-faild', ldm_time.total_seconds(), post_time.total_seconds(), result_status)
                print(f"B-rep rebuild failed due to exception: {e}")
                continue
    return succ  

def save_sampling_result(eval_args, iter, batch_idx, result, ldm_time,post_time, output_file):
    """
    保存采样结果为JSON文件。
    
    :param eval_args: dict，包含 'class_label' 等参数
    :param iter: int，迭代次数
    :param batch_idx: int，批次索引
    :param result: str，结果字符串
    :param time_interval: float，时间间隔（秒）
    :param output_file: str，JSON文件路径
    """
    # 创建当前记录
    unique_id = f"{eval_args['class_label']}_{iter}_batch_{batch_idx}"
    record = {
        "unique_id": unique_id,
        "result": result,
        "ldm_time": ldm_time,
        "post_time": post_time
    }
    
    # 如果文件存在，读取已有数据
    try:
        with open(output_file, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []  # 文件不存在则创建空列表
    
    # 追加记录
    data.append(record)
    
    # 保存到JSON文件
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4)  # 使用缩进格式化保存
    print(f"Record saved: {record}")

def update_fn(x,out_score,t,x_sde,x_rsde):
    size= len(out_score.shape)
    x_score=score_fn(out_score,t,x_sde)
    dt = -1. / x_sde.N
    
    if size==3:
        z_x = torch.randn_like(x)
        drift_x, diffusion_x = x_rsde.sde_score(x, t, x_score)
        x_mean = x + drift_x * dt
        x = x_mean + diffusion_x[:, None, None] * np.sqrt(-dt) * z_x

    else:
        z_x = torch.randn_like(x)
        z_x = torch.tril(z_x, -1)
        z_x = z_x + z_x.transpose(-1, -2)
        drift_x, diffusion_x = x_rsde.sde_score(x, t, x_score)

        x_mean = x + drift_x * dt
        x = x_mean + diffusion_x[:, None, None, None] * np.sqrt(-dt) * z_x

    return x,x_mean

from utils import *
if __name__ == "__main__":

    text2int_furniture_ori = {'bathtub':1, 
            'bed':2, 
            'bench':3, 
            'bookshelf':4,
            'cabinet':5, 
            'chair':6, 
            'couch':7, 
            'lamp':8, 
            'sofa':9, 
            'table':10
            }

    text2int_cadnet10_ori = {'AngleIron': 1, 
                         'FlatKey': 2, 
                         'Grooved_Pin': 3, 
                         'HookWrench': 4, 
                         'Key': 5, 
                         'Spring': 6,
                        'Steel': 7, 'TangentialKey': 8, 'Thrust_Ring': 9, 'Washer': 10}

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['abc', 'deepcad', 'furniture','cadnet'], default='abc', 
                        help="Choose between evaluation mode [abc/deepcad/furniture] (default: abc)")
    parser.add_argument("--model", type=str, choices=['brepgen', 'brepgd'], default='brepgen')
    args = get_args_ldm()
    args = parser.parse_args()    
    model=args.model
    if args.mode == 'furniture':
        text2int_ori=text2int_furniture_ori
    else:
        text2int_ori=text2int_cadnet10_ori


    if model=='brepgen':
        yaml_name='eval_config.yaml'
    elif model=='brepgd':
        yaml_name='eval_config_brepgd.yaml'
    

    with open(yaml_name, 'r') as file:
        config = yaml.safe_load(file)

    eval_args = config[args.mode]
    
    set_seed(eval_args['seed'])
    save_path=eval_args['save_folder']
    batch_size=eval_args['batch_size']
    result_status=save_path+'/result.json'
    builded=0
    all_build=0
    i=0
    while builded <3000:
        keys = list(text2int_ori.keys())
        random_key = random.choice(keys)
        eval_args['class_label']=random_key
        eval_args['save_folder']=save_path+'/'+random_key
        if args.mode=='deepcad':
            eval_args['save_folder']=save_path
        print(f'gen_class:{random_key}')
        i+=1
        if model=='brepgen':
            builded=builded+sample(eval_args,i,args.mode,result_status)
        else:
            builded=builded+sample_GBrep_noC(eval_args,i,args.mode,result_status)
        all_build+=batch_size
        print(f'Successful buided: {builded} / {all_build}')





        
        
