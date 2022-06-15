import sys
root = "/home/ytaima/code/dl-autowarp"
sys.path.insert(0, root)
sys.path.insert(0, "core")

import os
import torch
import tqdm as tq
import datasets
from raft import RAFT
import numpy as np
import matplotlib.pyplot as plt
from lib.flow_utils import write_flow
import random
from easydict import EasyDict

dsroot = root + "/data/warpsds"
args = EasyDict()
args.name = RAFT
args.restore_ckpt = "models/raft-asphere.pth"
args.small = False
args.mixed_precision = False
args.gpus = [0, 1]
model = RAFT(args)
state0 = torch.load(args.restore_ckpt)
state = {k.replace("module.", ""):v for k, v in state0.items()}
model.load_state_dict(state)
model.cuda();
model.eval();
testset = datasets.AsphereWarp(split="validation", crop=(1200, 1200))
testset_path = dsroot + "/validation.txt"
#ids = np.loadtxt(testset_path, dtype=int).tolist()
#ids = random.sample(ids, 100)
viz_predicted_dir = dsroot + "/viz_flows_predicted"
flows_predicted_dir = dsroot + "/flows_predicted"
os.makedirs(viz_predicted_dir, exist_ok=True)
os.makedirs(flows_predicted_dir, exist_ok=True)
for id_ in tq.tqdm(ids, desc = "Processing...", leave=False):
    image1, image2, flo, valid, extra_info = testset[id_]
    flows = model.forward(image1[None].cuda(), image2[None].cuda(), iters=1)
    f = flows[0].squeeze()
    mag_gt = torch.hypot(*flo)
    mag_gt = mag_gt.detach().cpu()
    mag = torch.hypot(*f)
    mag = mag.detach().cpu()
    flow = np.transpose(f.detach().cpu())
    flow_path = dsroot + f"/flows_predicted/{id_}.flo"
    write_flow(flow_path, flow)
    plt.figure(figsize=(18,18))
    plt.subplot(2,2,1)
    plt.axis("equal")
    plt.contour(mag, origin="upper")
    plt.title("prediction")
    plt.subplot(2,2,2)
    plt.contour(mag_gt, origin="upper")
    plt.axis("equal")
    plt.title(f"ground truth {id_}")
    plt.subplot(2,2,3)
    plt.imshow(mag)
    plt.title(f"prediction {id_}")
    plt.subplot(2,2,4)
    plt.imshow(mag_gt)
    plt.title(f"ground truth {id_}")
    viz_predicted_path = dsroot + f"/viz_flows_predicted/{id_}-mag.png"
    plt.savefig(viz_predicted_path)
    plt.figure(figsize=(18,9))
    plt.subplot(1,2,1)
    plt.title("satimage")
    plt.imshow(image1.numpy().astype(np.uint8).transpose(1,2,0))
    plt.subplot(1,2,2)
    plt.title("snapshot")
    plt.imshow(image2.numpy().astype(np.uint8).transpose(1,2,0))
    viz_predicted_path = dsroot + f"/viz_flows_predicted/{id_}-pairs.png"
    plt.savefig(viz_predicted_path)
    plt.close("all")
