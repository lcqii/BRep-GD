export CUDA_VISIBLE_DEVICES=3
python ldm.py --data /home/luochenqi/lcq/experiments/GDBG/data_process/deepcad_parsed \
    --list /home/luochenqi/lcq/experiments/GDBG/data_process/deepcad_split_6bit_filtered_balanced.pkl --option GEdgeZ \
    --surfvae /home/luochenqi/lcq/experiments/GDBG/checkpoints/deepcad_vae_surf.pt --edgevae /home/luochenqi/lcq/experiments/GDBG/checkpoints/deepcad_vae_edge.pt --gpu 3\
    --env deepcad_ldm_GEdgeZ --train_nepoch 1000 --test_nepoch 1 --save_nepoch 20 --batch_size 2 \
    --max_face 50 --max_edge 50 --bbox_scaled 1 --threshold 0.02