python -u train.py \
    --name raft-asphere \
    --stage asphere \
    --validation asphere \
    --gpus 0 1 \
    --num_steps 100000 \
    --batch_size 2 \
    --lr 0.0004 \
    --image_size 1024 1024 \
    --wdecay 0.00001 \
    --gamma=0.85 \
    --mixed_precision \
    --num-workers 24 | tee -a train.log
