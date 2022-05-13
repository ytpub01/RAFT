python -u train.py \
    --name raft-asphere \
    --stage asphere \
    --validation asphere \
    --batch_size 2 \
    --image_size 1152 1152 \
    --wdecay 0.00001 \
    --gamma 0.85 \
    --num_steps 40000 \
    --restore_ckpt models/raft-asphere.pth \
    | tee -a train.log
