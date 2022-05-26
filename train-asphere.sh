python -u train.py \
    --name raft-asphere \
    --stage asphere \
    --validation asphere \
    --batch_size 2 \
    --image_size 1152 1152 \
    --lr 0.0004 \
    --wdecay 0.00001 \
    --gamma 0.85 \
    --num_steps 150000 \
    --restore_ckpt models/raft-kitti.pth \
    | tee -a train.log
