#!/bin/bash\

### Train the Latent Diffusion Model ###
# --data_aug is optional
# max_face 30, max_edge 20 for deepcad 
# max_face 50, max_edge 30 for abc/furniture
# --surfvae refer to the surface vae weights 
# --edgevae refer to the edge vae weights 

# ### Training DeepCAD Latent Diffusion Model ###  
# python ldm.py --data data_process/deepcad_parsed \
#     --list data_process/deepcad_data_split_6bit.pkl --option surfpos --gpu 0 1 \
#     --env deepcad_ldm_surfpos --train_nepoch 3000 --test_nepoch 200 --save_nepoch 200 \
#     --max_face 30 --max_edge 20

# python ldm.py --data data_process/deepcad_parsed \
#     --list data_process/deepcad_data_split_6bit.pkl --option surfz \
#     --surfvae proj_log/deepcad_vae_surf.pt --gpu 0 1 \
#     --env deepcad_ldm_surfz --train_nepoch 3000 --batch_size 256 \
#     --max_face 30 --max_edge 20

# python ldm.py --data data_process/deepcad_parsed \
#     --list data_process/deepcad_data_split_6bit.pkl --option edgepos \
#     --surfvae proj_log/deepcad_vae_surf.pt --gpu 0 1 \
#     --env deepcad_ldm_edgepos --train_nepoch 1000 --batch_size 128 \
#     --max_face 30 --max_edge 20

# python ldm.py --data data_process/deepcad_parsed \
#     --list data_process/deepcad_data_split_6bit.pkl --option edgez \
#     --surfvae proj_log/deepcad_vae_surf.pt --edgevae proj_log/deepcad_vae_edge.pt --gpu 0 1 \
#     --env deepcad_ldm_edgez --train_nepoch 1000 --batch_size 128 \
#     --max_face 30 --max_edge 20


### Training ABC Latent Diffusion Model ###  
# python ldm.py --data data_process/abc_parsed \
#     --list data_process/abc_data_split_6bit.pkl --option surfpos --gpu 0 1 \
#     --env abc_ldm_surfpos --train_nepoch 1000 --test_nepoch 200 --save_nepoch 200 \
#     --max_face 50 --max_edge 30

# python ldm.py --data data_process/abc_parsed \
#     --list data_process/abc_data_split_6bit.pkl --option surfz \
#     --surfvae proj_log/abc_vae_surf.pt --gpu 0 1 \
#     --env abc_ldm_surfz --train_nepoch 1000 --batch_size 256 \
#     --max_face 50 --max_edge 30

# python ldm.py --data data_process/abc_parsed \
#     --list data_process/abc_data_split_6bit.pkl --option edgepos \
#     --surfvae proj_log/abc_vae_surf.pt --gpu 0 1 \
#     --env abc_ldm_edgepos --train_nepoch 300 --batch_size 64 \
#     --max_face 50 --max_edge 30

# python ldm.py --data data_process/abc_parsed \
#     --list data_process/abc_data_split_6bit.pkl --option edgez \
#     --surfvae proj_log/abc_vae_surf.pt --edgevae proj_log/abc_vae_edge.pt --gpu 0 1 \
#     --env abc_ldm_edgez --train_nepoch 300 --batch_size 64 \
#     --max_face 50 --max_edge 30


# ### Training Furniture Latent Diffusion Model (classifier-free) ###  
# python ldm.py --data data_process/furniture_parsed \
#     --list data_process/furniture_data_split_6bit.pkl --option surfpos --gpu 0 1 \
#     --env furniture_ldm_surfpos --train_nepoch 3000 --test_nepoch 200 --save_nepoch 200 \
#     --max_face 50 --max_edge 30 --cf

# python ldm.py --data data_process/furniture_parsed \
#     --list data_process/furniture_data_split_6bit.pkl --option surfz \
#     --surfvae proj_log/furniture_vae_surf.pt --gpu 0 1 \
#     --env furniture_ldm_surfz --train_nepoch 3000 --batch_size 256 \
#     --max_face 50 --max_edge 30 --cf

# python ldm.py --data /root/autodl-tmp/furniture_parsed \
#     --list /root/autodl-tmp/furniture_data_split_6bit_filtered.pkl --option edgepos \
#     --surfvae /root/autodl-tmp/furniture_vae_surf.pt --gpu 0 1 \
#     --env furniture_ldm_edgepos --train_nepoch 500 --save_nepoch 1 --batch_size 6 \
#     --max_face 50 --max_edge 50 --cf
#tensorboard --logdir=proj_log/furniture_ldm_edgez/tensorboard_train --host=127.0.0.1 --port=8899

# /root/autodl-tmp/furniture_split_6bit_filtered_balanced.pkl
# /root/autodl-tmp/furniture_data_split_6bit_filtered.pkl

# python ldm.py --data /root/autodl-tmp/furniture_parsed \
#     --list /root/autodl-tmp/furniture_split_6bit_filtered_balanced.pkl --option gedgez \
#     --surfvae /root/autodl-tmp/furniture_vae_surf.pt --edgevae /root/autodl-tmp/furniture_vae_edge.pt --gpu 0 1\
#     --env furniture_ldm_gedgez_g --train_nepoch 1000 --test_nepoch 5 --batch_size 96 \
#     --max_face 50 --max_edge 50 --bbox_scaled 1 --cf


# tensorboard --logdir=proj_log/furniture_ldm_edgez/tensorboard_train --host=127.0.0.1 --port=7777
python ldm.py --data /root/autodl-tmp/furniture_parsed \
    --list /root/autodl-tmp/furniture_split_6bit_filtered_balanced.pkl --option edgez \
    --surfvae /root/autodl-tmp/furniture_vae_surf.pt --edgevae /root/autodl-tmp/furniture_vae_edge.pt --gpu 0 1\
    --env furniture_ldm_edgez --train_nepoch 1000 --test_nepoch 5 --batch_size 48 \
    --max_face 50 --max_edge 50 --cf