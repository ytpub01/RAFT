import sys
import torch
import torch.nn as nn
import tqdm as tq
import datasets
from raft import RAFT
import numpy as np
from easydict import EasyDict
from utils.utils import center_crop
import os.path as osp
from lib.flow_utils import visualize_flow_file, write_flow

root = osp.join("home", "ytaima", "code", "dl-autowarp")
sys.path.insert(0, root)
sys.path.insert(0, "core")
dsroot = osp.join(root, "data", "warpsds")
args = EasyDict()
args.stage = "asphere"
args.restore_ckpt = osp.join(root, "ext", "RAFT", "models", "raft-asphere.pth")
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
testset_path = osp.join(dsroot, "validation.txt")
ids = np.loadtxt(testset_path, dtype=int).tolist()
num_ids = len(ids)
pe_global = 0
count = 0
ids_dict = {}
 
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
    pred_flow_path = osp.join(dsroot, "flows_predicted", f"{panoid}.flo")
    write_flow(pred_flow_path, pred_flow)
    assert pred_flow.shape[2] == 2, "u, v channels must be last"
    assert gt_flow.shape[2] == 2, "u, v channels must be last"
    crop = pred_flow.shape[:2]
    gt_flow = center_crop(gt_flow, crop, channels_first=False)
    gt_flow_u_min = np.abs(gt_flow[...,0]).min()
    gt_flow_v_min = np.abs(gt_flow[...,1]).min()
    gt_flow_u_mean = np.abs(gt_flow[...,0]).mean()
    gt_flow_v_mean = np.abs(gt_flow[...,1]).mean()
    gt_flow_mag = np.hypot(gt_flow[...,0], gt_flow[...,1])
    gt_flow_mag_min = gt_flow_mag.min()
    gt_flow_mag_max = gt_flow_mag.max()
    gt_flow_mag_mean = gt_flow_mag.mean()
    pixel_error = np.mean(np.sqrt(np.sum(np.square(pred_flow - gt_flow), axis=2)))
    if pixel_error < args.max_error:
        pe_global += pixel_error
        count += 1
    ids_dict[panoid] = [pixel_error, gt_flow_u_min, gt_flow_v_min, gt_flow_u_mean, gt_flow_v_mean,
                            gt_flow_mag_min, gt_flow_mag_max, gt_flow_mag_mean]
pe_global /= count
percent_bad = (len(ids) - count)/len(ids)*100
with open("predict_stats.txt", 'w') as f:
    f.write("panoid\tpixel_error\tgt_flow_u_min\tgt_flow_v_min\tgt_flow_u_mean\tgt_flow_v_mean\t"
                               "gt_flow_mag_min\tgt_flow_mag_max\tgt_flow_mag_mean\n")
    for k, v in ids_dict.items():
        f.write(str(k) + '\t' + "\t".join([str(val) for val in v])+'\n')
result=f"Mean square error for the {num_ids} validation set is {pe_global:.2f}, with {percent_bad:.2f} percent bad ids."
print(result)