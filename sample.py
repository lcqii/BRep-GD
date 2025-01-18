import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from network import *
from models.GraphBrep import *
from diffusers import DDPMScheduler, PNDMScheduler
import json
from OCC.Extend.DataExchange import write_stl_file, write_step_file
from utils import *
text2int_cadnet10 = {'uncond':0, 
                         'AngleIron': 1, 
                         'FlatKey': 2, 
                         'Grooved_Pin': 3, 
                         'HookWrench': 4, 
                         'Key': 5, 
                         'Spring': 6,
                        'Steel': 7, 'TangentialKey': 8, 'Thrust_Ring': 9, 'Washer': 10}


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


def sample_GBrep_noC(eval_args,iter,mode='furniture'):
    torch.cuda.empty_cache()
    # Inference configuration
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
        # class_label = torch.LongTensor([text2int[eval_args['class_label']]]*batch_size).cuda().reshape(-1,1) 
        w = 0.6
    else:
        class_label = None
    #print(class_label)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    

    surfPos_model = SurfPosNet(eval_args['use_cf'])
    surfPos_model.load_state_dict(torch.load(eval_args['surfpos_weight'],weights_only=True))  
    surfPos_model = surfPos_model.to(device).eval()

    surfZ_model = SurfZNet(eval_args['use_cf'])
    surfZ_model.load_state_dict(torch.load(eval_args['surfz_weight'],weights_only=True))
    surfZ_model = surfZ_model.to(device).eval()


    #GEvp_model = GEvpACNet(eval_args['use_cf'])
    #edgePos_model = GEdgePosNet_ATT(eval_args['use_cf'])
    #edgePos_model = GEdgePosNet_ATT_noC(eval_args['use_cf'])
    # edgePos_model = GEdgePosNet_noC(eval_args['use_cf'])
    #edgePos_model = GEdgePosNet_AGL_noC(eval_args['use_cf'])
    #edgePos_model = EdgePosNet_refine(eval_args['use_cf'])
    #edgePos_model = EdgePosNet_branch(eval_args['use_cf'])
    # edgeZ_model = EdgeZNet_ori(eval_args['use_cf'])
    # edgePos_model = EdgePosNet_ori(eval_args['use_cf'])
    # edgeZ_model = GNN_EZ(eval_args['use_cf'])
    # edgePos_model = GNN_EP(eval_args['use_cf'])
    # edgeZ_model = TRANS_EZ(eval_args['use_cf'])
    # edgePos_model = TRANS_EP(eval_args['use_cf'])
    edgeZ_model = CGTD_EZ(eval_args['use_cf'])
    edgePos_model = CLA_EP(eval_args['use_cf'])
    #edgePos_model = CGTD_EP(eval_args['use_cf'])
    edgeZ_model.load_state_dict(torch.load(eval_args['GEdgeZ_weight'],weights_only=True))
    edgePos_model.load_state_dict(torch.load(eval_args['GEdgePos_c_weight'],weights_only=True))
    #edgePos_model.load_state_dict(torch.load(eval_args['GEdgePos_weight'],weights_only=True))
    # edgePos_model = EdgePosNet(eval_args['use_cf'])
    # edgePos_model.load_state_dict(torch.load(eval_args['edgepos_weight']))
    # edgePos_model = GEdgePosNet_back(eval_args['use_cf'])
    # edgePos_model.load_state_dict(torch.load(eval_args['GEdgePos_weight']))
    edgePos_model = edgePos_model.to(device).eval()

    # edgeZ_model = EdgeZNet(eval_args['use_cf'])
    # edgeZ_model.load_state_dict(torch.load(eval_args['edgez_weight']))
    #edgeZ_model = GEdgeZNet_noC(eval_args['use_cf'])
    #edgeZ_model = GEdgeZNet(eval_args['use_cf'])
    #edgeZ_model = EdgeZNet_refine(eval_args['use_cf'])
    #edgeZ_model = CGTD_EZ(eval_args['use_cf'])
    #edgeZ_model = EdgeZNet_branch(eval_args['use_cf'])
    #edgeZ_model.load_state_dict(torch.load(eval_args['GEdgeZ_weight'],weights_only=True))
    edgeZ_model = edgeZ_model.to(device).eval()

    # edgeZ_model = EdgeZNet(eval_args['use_cf'])
    # edgeZ_model.load_state_dict(torch.load(eval_args['edgez_weight']))
    # edgeZ_model = edgeZ_model.to(device).eval()
    
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
    succ=0
    #torch.cuda.empty_cache()
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            ###########################################
            # STEP 1-1: generate the surface position #
            ###########################################

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
           
            # Late increase for ABC/DeepCAD (slightly more efficient)
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
           

            #######################################
            # STEP 1-2: remove duplicate surfaces #
            #######################################
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
            #torch.cuda.empty_cache()
            surfPos = torch.stack(surfPos_deduplicate).cuda()
            surfMask = torch.vstack(surfMask_deduplicate).cuda()
            surfPos=surfPos[:,:num_edges,:]
            surfMask=surfMask[:,:num_edges]

            edgeM=surfMask.unsqueeze(-1) * surfMask.unsqueeze(1)
            face_mask=(~surfMask).to(torch.int)
            edge_mask=face_mask.unsqueeze(-1) * face_mask.unsqueeze(1)
            edge_mask = torch.tril(edge_mask, -1) + torch.tril(edge_mask, -1).transpose(-1, -2)
            edge_mask=edge_mask.unsqueeze(-1)

            #################################
            # STEP 1-3:  generate surface z #
            #################################
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
            #torch.cuda.empty_cache()
            # ########################################
            # # STEP 2-1: generate the edge position #
            # ########################################
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
                # timesteps=timesteps.repeat(batch_size)
                # noise_pred = edgePos_model(edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
                edgePos = pndm_scheduler.step(noise_pred, t, edgePos).prev_sample
                edgePos = edgePos.permute(0, 3, 1, 2)
                edgePos=(torch.tril(edgePos, -1) + torch.tril(edgePos, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  

            ddpm_scheduler.set_timesteps(1000)  
            for t in tqdm(ddpm_scheduler.timesteps[-250:]):
                timesteps = t.reshape(-1).cuda()  
                #timesteps=timesteps.repeat(2*batch_size)
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
                # timesteps=timesteps.repeat(batch_size)
                # noise_pred = edgePos_model(edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
                edgePos = ddpm_scheduler.step(noise_pred, t, edgePos).prev_sample
                edgePos = edgePos.permute(0, 3, 1, 2)
                edgePos=(torch.tril(edgePos, -1) + torch.tril(edgePos, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  
            #edgeM=(adj>0.5).squeeze()
            #edgePos=edgePos[...,1:]*3.0
            #surfPos=surfPos * 3.0
            #edgePos=(edgePos+edgePos.transpose(1, 2))/2
            
            #edgePos = edgePos* edge_mask # set removed data to 0
            #edgePos=evp[:,:,:,:6]
            #edgeV=evp[:,:,:,6:].detach().cpu().numpy()

            #torch.cuda.empty_cache()
            edgeM=((edgePos<0.2).all(-1) & (edgePos>-0.2).all(-1))
            edge_mask=(~edgeM).float().unsqueeze(-1)
            ##############################
            # STEP 2-3: generate edge zv #
            ##############################   
            edgeZV= randn_tensor((batch_size,18,num_edges,num_edges)).cuda()
            edgeZV=(torch.tril(edgeZV, -1) + torch.tril(edgeZV, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask  
            pndm_scheduler.set_timesteps(200)   
            for t in tqdm(pndm_scheduler.timesteps):
                timesteps = t.reshape(-1).cuda()   
                # timesteps=timesteps.repeat(batch_size)
                # if class_label is not None:
                #     _surfZ_ = surfZ.repeat(2,1,1)
                #     _surfPos_ = surfPos.repeat(2,1,1)
                #     _edgePos_ = edgePos.repeat(2,1,1,1)
                #     _face_mask_ = face_mask.repeat(2,1)
                #     _edge_mask_ = edge_mask.repeat(2,1,1,1)
                #     _edgeZV_ = edgeZV.repeat(2,1,1,1)
                #     noise_pred = edgeZ_model(_edgeZV_, _edgePos_, _surfPos_, _surfZ_, timesteps, _face_mask_,_edge_mask_, class_label)
                #     noise_pred = noise_pred[:batch_size] * (1+w) - noise_pred[batch_size:] * w
                # else:
                #     noise_pred = edgeZ_model(edgeZV, edgePos, surfPos, surfZ, timesteps, face_mask,edge_mask, class_label)
                noise_pred = edgeZ_model(edgeZV,edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
                timesteps=timesteps.repeat(batch_size)
                edgeZV = pndm_scheduler.step(noise_pred, t, edgeZV).prev_sample
                edgeZV = edgeZV.permute(0, 3, 1, 2)
                edgeZV=(torch.tril(edgeZV, -1) + torch.tril(edgeZV, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask 

                # noise_pred = edgeZ_model(edgeZV,edgePos,surfPos, surfZ,timesteps,face_mask,edge_mask)
                # noise_pred = noise_pred.permute(0, 3, 1, 2)
                # noise_pred=(torch.tril(noise_pred, -1) + torch.tril(noise_pred, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask 
                # edgeZV = ddpm_scheduler.step(noise_pred, t, edgeZV).prev_sample
                # edgeZV = edgeZV.permute(0, 3, 1, 2)
                # edgeZV=(torch.tril(edgeZV, -1) + torch.tril(edgeZV, -1).transpose(-1, -2)).permute(0, 2, 3, 1)* edge_mask   
            edgePos=edgePos*3
            surfPos=surfPos*3
           
            
            edgeZV=(edgeZV+edgeZV.transpose(1, 2))/2
            edge_z = edgeZV[:,:,:,:12]
            edgeV = edgeZV[:,:,:,12:].detach()
            # edgeM_P=(edgePos<0.5).all(-1) & (edgePos>-0.5).all(-1)
            # edgeM_Z=(edge_z<0.3).all(-1) & (edge_z>-0.3).all(-1)
            # edgeM_V=(edgeV<0.5).all(-1) & (edgeV>-0.5).all(-1)
            
            #edgeM=edgeM_P & edgeM_Z & edgeM_V

            edgePos[edgeM] = 0 # set removed data to 0
            edge_z[edgeM] = 0 # set removed data to 0
            edgeV[edgeM] = 0 # set removed data to 0
            edgeV = edgeV.cpu().numpy()
            
            
            # Decode the surfaces
            surf_ncs = surf_vae(surfZ.unflatten(-1,torch.Size([16,3])).flatten(0,1).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
            surf_ncs = surf_ncs.permute(0,2,3,1).unflatten(0, torch.Size([batch_size, num_edges])).detach().cpu().numpy()
            
            # Decode the edges
            edge_ncs = edge_vae(edge_z.unflatten(-1,torch.Size([4,3])).reshape(-1,4,3).permute(0,2,1))
            edge_ncs = edge_ncs.permute(0,2,1).reshape(batch_size, num_edges, num_edges, 32, 3).detach().cpu().numpy()
            ldm_time = (end_time - start_time)/batch_size

            surfMask=edgeM.all(dim=-1)
            edge_mask = edgeM.detach().cpu().numpy()     
            #edge_mask = (edge_mask>0.5).squeeze().detach().cpu().numpy()     
            edge_pos = edgePos.detach().cpu().numpy() / 3.0
            surfPos = surfPos.detach().cpu().numpy()  / 3.0
            # edge_pos = edgePos.detach().cpu().numpy()
            # surfPos = surfPos.detach().cpu().numpy()
            #edge_ncs2= edgeZV[:,:,:,12:].reshape(batch_size, num_surfaces, num_surfaces, 2, 3).detach().cpu().numpy()
            edge_ncs2= edgeV.reshape(batch_size, num_edges, num_edges, 2, 3)
    #torch.cuda.empty_cache()
    
    #############################################
    ### STEP 3: Post-process (per-single CAD) ###
    #############################################
    for batch_idx in range(batch_size):
        # Per cad (not including invalid faces)
        surfMask_cad = surfMask[batch_idx].detach().cpu().numpy()
        edge_mask_cad = edge_mask[batch_idx][~surfMask_cad]
        edge_pos_cad = edge_pos[batch_idx][~surfMask_cad]
        edge_ncs_cad = edge_ncs[batch_idx][~surfMask_cad]
        edgeV_cad = edgeV[batch_idx][~surfMask_cad]
        edge_z_cad = edge_z[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()[~edge_mask_cad]
        surf_z_cad = surfZ[batch_idx][~surfMask[batch_idx]].detach().cpu().numpy()
        surf_pos_cad = surfPos[batch_idx][~surfMask_cad]

        # Retrieve vertices based on edge start/end
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
        
        ### 3-1: Detect shared vertices ###
        try:
            unique_vertices, new_vertex_dict = detect_shared_vertex2(edgeV_cad, edge_mask_cad, edgeV_bbox)
        except Exception as e:
            print('Vertex detection failed...')
            continue
        
        ### 3-2: Detect shared edges ###
        try:
            #unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj = detect_shared_edge(unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, z_threshold, edge_mask_cad)
            unique_faces, unique_edges, FaceEdgeAdj, EdgeVertexAdj = detect_shared_edge2(unique_vertices, new_vertex_dict, edge_z_cad, surf_z_cad, edge_mask_cad)
        except Exception as e:

            print('Edge detection failed...')
            continue
        
        # Decode unique faces / edges
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                surf_ncs_cad = surf_vae(torch.FloatTensor(unique_faces).cuda().unflatten(-1,torch.Size([16,3])).permute(0,2,1).unflatten(-1,torch.Size([4,4])))
                surf_ncs_cad = surf_ncs_cad.permute(0,2,3,1).detach().cpu().numpy()
                edge_ncs_cad = edge_vae(torch.FloatTensor(unique_edges).cuda().unflatten(-1,torch.Size([4,3])).permute(0,2,1))
                edge_ncs_cad = edge_ncs_cad.permute(0,2,1).detach().cpu().numpy()

        #### 3-3: Joint Optimize ###
        num_edge = len(edge_ncs_cad)
        num_surf = len(surf_ncs_cad)
        surf_wcs, edge_wcs = joint_optimize(surf_ncs_cad, edge_ncs_cad, surf_pos_cad, unique_vertices, EdgeVertexAdj, FaceEdgeAdj, num_edge, num_surf)
        
        #### 3-4: Build the B-rep ###
        with ThreadPoolExecutor() as executor:
            future = executor.submit(construct_brep, surf_wcs, edge_wcs, FaceEdgeAdj, EdgeVertexAdj)
            try:
                # 设置任务超时时间为 5 秒
                solid = future.result(timeout=10)
                write_step_file(solid, f'{save_folder}/{iter}_{batch_idx}.step')
                write_stl_file(solid, f'{save_folder}/{iter}_{batch_idx}.stl', linear_deflection=0.001, angular_deflection=0.5)
                #save_sampling_result(eval_args, iter, batch_idx, 'build-success', ldm_time.total_seconds(), post_time.total_seconds(), result_status)
                #print("B-rep rebuild successful.")
            
            except TimeoutError:
                # 捕获超时异常
             
                print("B-rep rebuild timed out.")
                continue
            
            except Exception as e:
                # 捕获其他异常
                print(f"B-rep rebuild failed due to exception: {e}")
                continue

    return succ
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=['abc', 'deepcad', 'furniture'], default='abc', 
                        help="Choose between evaluation mode [abc/deepcad/furniture] (default: abc)")
    args = parser.parse_args()    

    # Load evaluation config 
    with open('eval_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    eval_args = config[args.mode]
    iter=0
    while(True):
        sample_GBrep_noC(eval_args,iter)
        iter+=1