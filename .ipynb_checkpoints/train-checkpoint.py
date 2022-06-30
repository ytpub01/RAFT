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
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None
        self.examples = examples
        self.points = {}
    
    @torch.no_grad()
    def plot_examples(self):
        assert self.examples is not None, "set examples before plotting them"
        image1, image2, flow, valid, ids = [x.cuda() for x in self.examples]
        old_mode = self.model.training
        self.model.eval()
        _, flow_predictions = self.model(image1, image2, iters=args.iters, test_mode=True)
        self.model.train(old_mode)
        for id in ids:
            if id not in self.points:
                interleaved = np.loadtxt(f"/home/ytaima/warpsds/viz_preds/{id}-pts.txt")
                sat_points = interleaved[::2]
                
                # TODO: self.points[id] = 

    def _print_training_status(self, image1, image2, extra_info):
        training_str = "[{:6d}, lr={:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ""
        for k, v in self.running_loss.items(): metrics_str += f' {k}={v/SUM_FREQ:<10.4f}'
        # print the training status
        tq.tqdm.write(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter()
        
        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics, image1, image2, extra_info):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            self.running_loss[key] += metrics[key]
            
        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status(image1, image2, extra_info)
            self.running_loss = {}

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
    if args.restore_ckpt is not None:
        # also load state_dict for scheduler, optimizer and scaler
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
    model.cuda()
    model.train()
    if args.freeze_bn:
        model.module.freeze_bn()
    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)
    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)
    total_progress = tq.tqdm(desc='Total', total=args.num_steps)
    optimizer.step() # must execute before scheduler
    should_keep_training = True
    while should_keep_training:
        loop = tq.tqdm(train_loader, desc="Training", leave=False)
        for data_blob in loop:
            #Train Step
            optimizer.zero_grad()
            image1, image2, flow, valid, extra_info = [x.cuda() for x in data_blob]
            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)
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
            loop.set_postfix({"L":loss.item(), "I1":extra_info[0].item(), "I2":extra_info[1].item()})
            logger.push(metrics, image1, image2, extra_info)
            # Validation 
            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                # TODO: save optimizer and scheduler, and scalar
                torch.save(model.state_dict(), PATH)
                results = {}
                for val_dataset in args.validation:
                    if val_dataset == 'asphere':
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
    torch.save(model.state_dict(), PATH)
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
