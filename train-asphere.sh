python -u train.py \
    --name raft-asphere \
    --stage asphere \
    --validation asphere \
    --image_size 1152 1152 \
    --lr 0.0004 \
    --wdecay 0.00001 \
    --gamma 0.85 \
    --batch_size 2 \
    --num_steps 110000 \
    --restore_ckpt models/raft-asphere.pth \
    | tee -a train.log
#    > train_dbg.log 2>&1
