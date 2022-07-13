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
import matplotlib.pyplot as plt
from easydict import EasyDict
import viz_preds
#from torchvision.transforms.functional import to_pil_image

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
#ids = random.sample(ids, 100)
#ids = [339] # To debug
for id_ in tq.tqdm(range(num_ids), desc = "Processing...", leave=False):
    satimage, snapshot, gt_flow, valid, panoid = testset[id_]
    panoid = panoid.item()
    # save file to debug
    #satimage_2 = to_pil_image(satimage/255)
    #satimage_2.save(root + "/14790_satimage_scr.jpg")
    _, pred_flow = model(image1=satimage[None].cuda(),
                                image2=snapshot[None].cuda(),
                                iters=args.iters, 
                                test_mode=True)
    pred_flow = pred_flow.squeeze(0).detach().cpu()
    params = dict(id_=panoid, 
              root="data/warpsds",
              pred_flow=pred_flow.permute(1,2,0).numpy(),
              gt_flow=gt_flow.permute(1,2,0).numpy(),
              satimage=satimage.permute(1,2,0).to(torch.uint8).numpy(),
              snapshot=snapshot.permute(1,2,0).to(torch.uint8).numpy())

    fig = viz_preds.plot_pts_id(**params);
    plt.figure(fig.number)
    viz_predicted_path = dsroot + f"/viz_preds/{panoid}-pts.png"
    plt.savefig(viz_predicted_path)
    plt.close("all")
