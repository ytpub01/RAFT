# Need 0 workers to debug
ipython --pdb train.py \
    --name raft-asphere \
    --stage asphere \
    --validation asphere \
    --gpus 0 1 \
    --num_steps 100000 \
    --batch_size 10 \
    --lr 0.0004 \
    --image_size 512 512 \
    --wdecay 0.00001 \
    --gamma=0.85 \
    --num-workers 0 \
    --mixed_precision 
