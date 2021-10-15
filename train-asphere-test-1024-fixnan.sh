#
# I noticed that BN was frozen becaus of some silly code in the RAFT source
python -u train.py \
    --name raft-asphere-test-1024-fixnan \
    --stage asphere \
    --restore_ckpt checkpoints/raft-asphere-test-1024.pth \
    --validation asphere \
    --gpus 0 1 \
    --num_steps 100000 \
    --batch_size 2 \
    --lr 0.0004 \
    --image_size 1024 1024 \
    --wdecay 0.00001 \
    --gamma=0.85 \
    --num-workers 0 
#    --num-workers 25 | tee -a train-test.log
