# CUDA_VISIBLE_DEVICES=0 python sample.py --mode cadnet --model brepgd
mkdir -p logs/sample_cadnet 

CUDA_VISIBLE_DEVICES=3 nohup python sample.py \
    --mode cadnet \
    --model brepgd \
    > logs/sample_cadnet/sample_cadnet_$(date +%Y%m%d_%H%M%S).log 2>&1 &
