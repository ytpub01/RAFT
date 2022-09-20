import argparse
import sys
import os.path as osp
from os.path import abspath

root = abspath(osp.join("home", "ytaima", "code", "dl-autowarp"))
sys.path.insert(0, root)
#sys.path.insert(0, "core")
from lib.flow_utils import write_flow

import torch
import torch.nn as nn
import tqdm as tq
from raft import RAFT
import numpy as np
import matplotlib.pyplot as plt
from lib.viz_utils import plot_pts_id
from core.datasets import AsphereWarp

if __name__ == "__main__":
    dsroot = osp.join(root, "data", "warpsds")
    ckpt_path = osp.join(root, "ext", "RAFT", "model", "raft-asphere.pth")
    parser =argparse.ArgumentParser()
    parser.add_argument("--root", default=dsroot)
    parser.add_argument("--stage", default="asphere")
    parser.add_argument("--restore_ckpt", default=ckpt_path)
    parser.add_argument("--image_size", default=(1056, 1056))
    parser.add_argument("--batch_size", default=2)
    parser.add_argument("--workers", default=24)
    parser.add_argument("--small", default=False)
    parser.add_argument("--gpus", default=[0,1])
    parser.add_argument("--iters", default=24)
    parser.add_argument("--mixed_precision", default=False)
    parser.add_argument("--split", default="training")
    parser.add_argument('--id', nargs='+', type=int, help='ID of the pano', default=[])
    args = parser.parse_args()

    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    state = torch.load(args.restore_ckpt)
    model.load_state_dict(state, strict=False)
    torch.no_grad()
    model.cuda();
    model.eval();
    testset = AsphereWarp(root=args.root, split=args.split, crop=args.image_size)
    ids = args.id
    if len(ids) == 0:
        testset_path = osp.join(dsroot, f"{args.split}.txt")
        ids = np.loadtxt(testset_path, dtype=int).tolist()
    num_ids = len(ids)
    for id_ in tq.tqdm(ids, desc = "Processing...", leave=False):
        idx = testset.extra_info.index(id_)
        satimage, snapshot, gt_flow, valid, panoid = testset[idx]
        assert panoid == id_, "panoid from dataset and loop must be equal"
        # save file to debug
        # satimage_2 = to_pil_image(satimage/255)
        # satimage_2.save(osp.join(root, str(panoid)) + "_satimage_scr.jpg")
        _, pred_flow = model(image1=satimage[None].cuda(),
                                    image2=snapshot[None].cuda(),
                                    iters=args.iters, 
                                    test_mode=True)
        pred_flow = pred_flow.squeeze(0).detach().cpu()
        pred_flow_path = osp.join(args.dsroot, "flows_predicted", f"{panoid}.flo")
        write_flow(pred_flow_path, pred_flow)
        params = dict(id_=panoid,
                pred_flow=pred_flow.permute(1,2,0).numpy(),
                gt_flow=gt_flow.permute(1,2,0).numpy(),
                satimage=satimage.permute(1,2,0).to(torch.uint8).numpy(),
                snapshot=snapshot.permute(1,2,0).to(torch.uint8).numpy()
                )
        fig = plot_pts_id(**params);