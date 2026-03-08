export CUDA_VISIBLE_DEVICES=3
python ldm.py --data /home/luochenqi/lcq/experiments/GDBG/data_process/deepcad_parsed \
    --list /home/luochenqi/lcq/experiments/GDBG/data_process/deepcad_split_6bit_filtered_balanced.pkl --option surfpos --gpu 3 \
    --env deepcad_ldm_surfpos --train_nepoch 3000 --batch_size 4 --test_nepoch 1 --save_nepoch 10 \
    --max_face 50 --max_edge 50


