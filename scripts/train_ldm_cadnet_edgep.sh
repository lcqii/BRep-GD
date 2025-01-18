python ldm.py --data /root/autodl-tmp/cadnet40v2_parsed \
    --list /root/autodl-tmp/cadnet40v2_split_6bit_filtered.pkl --option gedgepos \
    --surfvae /root/autodl-tmp/cadnet40v2_vae_surf.pt  --gpu 0 1 \
    --env cadnet40v2_ldm_edgepos_branchf --train_nepoch 1500 --test_nepoch 5 --save_nepoch 100 --batch_size 32 \
    --max_face 50 --max_edge 50 --bbox_scaled 1 --threshold 0.02 --cf
