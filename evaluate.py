import sys
sys.path.append('core')
import argparse
import numpy as np
import torch
import torch.utils.data
import datasets
from raft import RAFT
import tqdm as tq

@torch.no_grad()
def validate_asphere(model, iters=24):
    """ Perform evaluation on the ASphere (test) split """
    model.eval()
    epe_list = []
    val_dataset = datasets.AsphereWarp(split='validation', crop=(1000,1000))
    for i in tq.trange(len(val_dataset), desc="Validating"):
        #TODO - mask out the valid flow vectors
        image1, image2, flow_gt, valid, extra_info = val_dataset[i]
        _, flow_pr = model(image1[None].cuda(), image2[None].cuda(), iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        valid = (valid > 0.5).view(-1)
        epe = epe.view(-1)
        epe_list.append(epe[valid].numpy())
    epes = np.concatenate(epe_list)
    epe = np.mean(epes)
    tq.tqdm.write(f"Validation ASphere EPE: {epe},  out of {len(epes)} measurments")
    return {'validation': epe}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()
    with torch.no_grad():
        if args.dataset == 'asphere':
            validate_asphere(model.module)