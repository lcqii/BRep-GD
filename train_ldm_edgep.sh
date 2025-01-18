python ldm.py --data /root/autodl-tmp/furniture_parsed \
    --list /root/autodl-tmp/furniture_split_6bit_filtered_balanced.pkl --option edgepos \
    --surfvae /root/autodl-tmp/furniture_vae_surf.pt --gpu 0 1 \
    --env furniture_ldm_edgepos --train_nepoch 1000 --test_nepoch 5 --batch_size 72 \
    --max_face 50 --max_edge 50 --cf


