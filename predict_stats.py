import sys
root = "/home/ytaima/code/dl-autowarp"
sys.path.insert(0, root)
sys.path.insert(0, "core")

import torch
import torch.nn as nn
import tqdm as tq
import datasets
from raft import RAFT
import numpy as np
from easydict import EasyDict
from utils.utils import center_crop
import csv

dsroot = root + "/data/warpsds"
args = EasyDict()
args.stage = "asphere"
args.restore_ckpt = root + "/ext/RAFT/models/raft-asphere.pth"
args.image_size= (1056, 1056)
args.batch_size = 2
args.workers = 24
args.small = False
args.gpus = [0, 1]
args.iters = 12
args.mixed_precision = False
args.max_error = 25
model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
state = torch.load(args.restore_ckpt)
model.load_state_dict(state, strict=False)
torch.no_grad()
model.cuda();
model.eval();
testset = datasets.AsphereWarp(split="validation", crop=args.image_size)
testset_path = dsroot + "/validation.txt"
ids = np.loadtxt(testset_path, dtype=int).tolist()
num_ids = len(ids)
pe_global = 0
count = 1
bad_ids = {}

with open("predict_stats.txt", 'w') as f: 
    for id_ in tq.tqdm(range(num_ids), desc = "Processing...", leave=False):
        satimage, snapshot, gt_flow, valid, panoid = testset[id_]
        panoid = panoid.item()
        _, pred_flow = model(image1=satimage[None].cuda(),
                                    image2=snapshot[None].cuda(),
                                    iters=args.iters, 
                                    test_mode=True)
        pred_flow = pred_flow.squeeze(0).detach().cpu()
        pred_flow = pred_flow.permute(1,2,0).numpy()
        gt_flow = gt_flow.permute(1,2,0).numpy()
        assert pred_flow.shape[2] == 2, "u, v channels must be last"
        assert gt_flow.shape[2] == 2, "u, v channels must be last"
        crop = pred_flow.shape[:2]
        gt_flow = center_crop(gt_flow, crop, channels_first=False)
        pixel_error = np.mean(np.sqrt(np.sum(np.square(pred_flow - gt_flow), axis=2)))
        if pixel_error < args.max_error:
            pe_global += pixel_error
            count += 1
        else:
            bad_ids[panoid] = pixel_error
        f.write(f"{panoid} {pixel_error:.2f}\n")
    pe_global /= count
    f.write(f"{pe_global:.2f}\n")
percent_bad = len(bad_ids)/len(ids)*100
bad_ids_sorted = dict(sorted(bad_ids.items(), key=lambda x:x[1], reverse=True))
with open("predict_bad_ids.txt", 'w') as f:
    for k, v in bad_ids_sorted.items():
        f.write(f"{k}\t{v}\n")
result=f"Mean square error for validation set is {pe_global:.2f}, with {percent_bad:.2f} percent bad ids."
print(result)