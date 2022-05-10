python -u train.py \
    --name raft-asphere \
    --stage asphere \
    --validation asphere \
    --restore_ckpt models/raft-kitti.pth \
    --batch_size 4 \
    --image_size 896 896 \
    --lr 0.0004 \
    --wdecay 0.00001 \
    --gamma 0.85 \
    | tee -a train.log
