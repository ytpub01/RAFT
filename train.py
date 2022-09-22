from __future__ import print_function, division
import sys
sys.path.append('core')
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from raft import RAFT
import evaluate
import datasets
from torch.cuda.amp import GradScaler
import tqdm as tq

# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000

DEVICE = torch.device("cuda")

def _safe_mean(a):
    """
        return zero instead of NaN for empty a
    """
    if len(a):
        return a.mean().item()
    return 0.

def sequence_loss(flow_preds, flow_gt, valid, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    # exclude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        if torch.isnan(flow_preds[i].abs()).any(): tq.tqdm.write("prediction is NaN")
        if torch.isnan(flow_gt.abs()).all(): tq.tqdm.write("input is NaN")
        i_loss = (flow_preds[i] - flow_gt).abs()
        i_loss[torch.isnan(i_loss)] = 0.
        valid_loss = valid[:, None] * i_loss
        flow_loss += i_weight * valid_loss.mean()
    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]  #same as epe.flatten()
    metrics = {
        'epe': _safe_mean(epe), # only counts valid pixels in mean due to previous line
        '3px': _safe_mean((epe < 3).float()),
        '5px': _safe_mean((epe < 5).float()),
        '10px': _safe_mean((epe < 10).float()),
    }
    return flow_loss, metrics

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler 

class Logger:
    def __init__(self, model, scheduler, examples=None):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 1
        self.running_loss = {}
        self.writer = None
        self.examples = examples
        self.points = {}
    
    @torch.no_grad()

    def _print_training_status(self):
        training_str = "[{:6d}, lr={:10.7f}] ".format(self.total_steps, self.scheduler.get_last_lr()[0])
        metrics_str = ""
        for k, v in self.running_loss.items(): metrics_str += f' {k}={v/SUM_FREQ:<10.4f}'
        # print the training status
        tq.tqdm.write(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()
        
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]
        result = self.running_loss['epe']/SUM_FREQ            
        if self.total_steps > 0 and self.total_steps % SUM_FREQ == 0:
            self._print_training_status()
            self.running_loss = {}
        self.total_steps += 1
        return result

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def train(args):
    torch.autograd.set_detect_anomaly(True)
    model = nn.DataParallel(RAFT(args), device_ids=args.gpus)
    tq.tqdm.write(f"Training with:\nlr={args.lr}, batch_size={args.batch_size}, image_size={args.image_size}")
    tq.tqdm.write(f"")
    tq.tqdm.write("Parameter Count: %d" % count_parameters(model))
    ckpt = None
    if args.restore_ckpt is not None:
        ckpt = torch.load(args.restore_ckpt, map_location=DEVICE)
        # also load state_dict for scheduler, optimizer and scaler
        if "model_state_dict" in ckpt:
            model.module.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.module.load_state_dict(ckpt, strict=False)
    model.to(DEVICE)
    model.train()
    if args.freeze_bn:
        model.module.freeze_bn()
    optimizer, scheduler = fetch_optimizer(args, model)
    scaler = GradScaler(enabled=args.mixed_precision)
    if ckpt is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if ckpt is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ckpt is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    train_loader = datasets.fetch_dataloader(args)
    logger = Logger(model, scheduler)
    lowest_epe = float('inf')
    total_steps = 1
    total_progress = tq.tqdm(desc='Total', initial=1, total=args.num_steps)
    optimizer.step() # must execute before scheduler
    should_keep_training = True
    while should_keep_training:
        loop = tq.tqdm(train_loader, initial=1, desc="Training", leave=False)
        for data_blob in loop:
            #Train Step
            optimizer.zero_grad()
            image1, image2, flow, valid, extra_info = [x for x in data_blob]
            image1, image2, flow, valid  = image1.to(DEVICE), image2.to(DEVICE), flow.to(DEVICE), valid.to(DEVICE)
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).to(DEVICE)).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).to(DEVICE)).clamp(0.0, 255.0)
            flow_predictions = model(image1, image2, iters=args.iters)            
            loss, metrics = sequence_loss(flow_predictions, flow, valid, args.gamma)
            if torch.isnan(loss):
                tq.tqdm.write("ERROR: Loss is NaN")
            elif np.isnan(metrics['epe']):
                tq.tqdm.write("ERROR: epe is NaN")
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
            error = loss.item()
            if error > 100:
                tq.tqdm.write(f"Large error > 100: {error:.4f} for ids {extra_info[0]},{extra_info[1]}")
            loop.set_postfix({"L":loss.item(), "im1":extra_info[0], "im2":extra_info[1]})
            epe = logger.push(metrics)
            # Best model
            if total_steps > 0 and total_steps % SUM_FREQ == 0:
                if epe < lowest_epe:
                    lowest_epe = epe
                    PATH = f"checkpoints/{args.name}_best.pth"
                    torch.save({"loss": epe,
                                "model_state_dict": model.module.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "scaler": scaler.state_dict()}, PATH)
                    tq.tqdm.write(f"At iteration {total_steps} saved best model with error {epe:.2f}")

            # Validation 
            if total_steps > 0 and total_steps % VAL_FREQ == 0:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps, args.name)
                # save optimizer and scheduler, and scaler
                torch.save({"loss": epe,
                            "model_state_dict": model.module.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "scaler": scaler.state_dict()}, PATH)
                results = {}
                for val_dataset in args.validation:
                    if val_dataset == "asphere":
                        results.update(evaluate.validate_asphere(model.module))
                logger.write_dict(results)            
                model.train()
                if args.freeze_bn:
                   model.module.freeze_bn()
            total_steps += 1
            total_progress.update(1)
            if total_steps > args.num_steps:
                should_keep_training = False
                break
    total_progress.close()
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save({"loss": epe,
                "model_state_dict": model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict()}, PATH)
    return PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft', help="name your experiment")
    parser.add_argument('--freeze_bn', action='store_true', help="Prevent batch-normalization from updating during training") 
    parser.add_argument('--stage', help="determines which dataset to use for training") 
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--workers', type=int, help="the number of workers to load data", default=os.cpu_count())
    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--image_size', type=int, nargs='+', default=[512, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0,1])
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    parser.add_argument('--add_noise', action='store_true')
    args = parser.parse_args()
    torch.manual_seed(1234)
    np.random.seed(1234)
    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    train(args)
