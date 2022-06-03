python -u train.py \
    --name raft-asphere \
    --stage asphere \
    --validation asphere \
    --image_size 768 768 \
    --lr 0.0004 \
    --wdecay 0.00001 \
    --gamma 0.85 \
    --num_steps 100000 \
    --restore_ckpt models/raft-asphere.pth \
    | tee -a train.log
#    > train_dbg.log 2>&1
