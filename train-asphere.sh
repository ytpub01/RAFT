python -u train.py \
    --name raft-asphere \
    --stage asphere \
    --validation asphere \
    --restore_ckpt models/raft-kitti.pth \
    --batch_size 2 \
    --image_size 1120 1120 \
    --lr 0.0004 \
    --wdecay 0.00001 \
    --gamma 0.85 \
    | tee -a train.log
