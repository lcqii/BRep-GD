python ldm.py --data /root/autodl-tmp/deepcad_parsed \
    --list /root/autodl-tmp/deepcad_data_split_6bit_filtered.pkl --option surfpos --gpu 0 1 \
    --env deepcad_ldm_surfpos_filtered_balanced --train_nepoch 3000 --batch_size 64 --test_nepoch 10 --save_nepoch 100 \
    --max_face 50 --max_edge 50
