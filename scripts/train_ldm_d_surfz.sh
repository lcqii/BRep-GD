python ldm.py --data /root/autodl-tmp/deepcad_parsed \
    --list /root/autodl-tmp/deepcad_data_split_6bit_filtered.pkl --option surfz \
    --surfvae /root/autodl-tmp/deepcad_vae_surf.pt --gpu 0 1 \
    --env deepcad_ldm_surfz_filtered --train_nepoch 3000 --test_nepoch 10 --save_nepoch 500 --batch_size 1024 \
    --max_face 50 --max_edge 50

