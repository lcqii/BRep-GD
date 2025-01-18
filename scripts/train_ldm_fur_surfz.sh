python ldm.py --data /root/autodl-tmp/furniture_parsed \
    --list /root/autodl-tmp/furniture_split_6bit_filtered_balanced.pkl --option surfz \
    --surfvae /root/autodl-tmp/furniture_vae_surf.pt --gpu 0 1 \
    --env furniture_ldm_surfz_filtered_balanced --train_nepoch 3000 --test_nepoch 10 --save_nepoch 500 --batch_size 64 \
    --max_face 50 --max_edge 50 --cf
