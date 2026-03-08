export CUDA_VISIBLE_DEVICES=3
python ldm.py --data /home/luochenqi/lcq/experiments/GDBG/data_process/deepcad_parsed \
    --list /home/luochenqi/lcq/experiments/GDBG/data_process/deepcad_split_6bit_filtered_balanced.pkl --option surfz \
    --surfvae /home/luochenqi/lcq/experiments/GDBG/checkpoints/deepcad_vae_surf.pt --gpu 3 \
    --env deepcad_ldm_surfZ --train_nepoch 3000 --batch_size 4 --test_nepoch 20 --save_nepoch 200 \
    --max_face 50 --max_edge 50 --threshold 0.02
