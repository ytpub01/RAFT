# Data loading based on https://github.com/NVIDIA/flownet2-pytorch
import numpy as np
import torch
import torch.utils.data as data
import random
import os.path as osp
import traceback
from utils import frame_utils
from lib.snapshot import center_crop
from utils.augmentor import FlowAugmentor, SparseFlowAugmentor

class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False):
        self.augmentor = None
        self.sparse = sparse
        if aug_params is not None:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)
        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.mask_list = []
        self.extra_info = []

    def __getitem__(self, index):
        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True
        index = index % len(self.image_list)
        flow, valid = self.read_flow(index, self.sparse)
        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])
        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        if not self.is_test and self.augmentor is not None:
            try:
                if self.sparse:
                    img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
                else:
                    img1, img2, flow = self.augmentor(img1, img2, flow)
            except Exception as e:
                id_ = self.image_list[index][0].split('/')[-1].split('.')[0]
                print(f"Skipping id {id_} with image size {img1.shape} due to an error {e} : {traceback.format_exc()}")
                raise
        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        return img1, img2, flow, valid.float(), self.extra_info[index]

    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
        
    def __len__(self):
        return len(self.image_list)
        
class AsphereWarp(FlowDataset):
    def __init__(self, aug_params=None, split='training', root='data/asphere', crop=None, sparse=True):
        """Dataset for ASphere flows
        Args:
            aug_params ([type], optional): Constructor arguments to Augmentor.
            split (str, optional): Data split. Defaults to 'training'.
            root (str, optional): Root folder of the dataset. Defaults to 'data/asphere'.
            crop ([type], optional): The size to crop image to. Defaults to None.
        NOTE:  The 'crop' argument is done here after standard augmentation, but it shouldn't be. 
        TODO: Fix augmentor to get rid of nonconfigurable/magic parameters and remove 'crop' argument. 
        """
        super(AsphereWarp, self).__init__(aug_params, sparse)
        if split == 'test' or split == 'test_tiles':
            self.is_test = True
        ids = np.loadtxt(osp.join(root, f"{split}.txt"), ndmin=1, dtype=str).tolist()
        if not self.is_test:
            flows = [osp.join(root, "flows", f"{id}.flo") for id in ids]
            masks = [osp.join(root, "masks", f"{id}.npz") for id in ids]
            self.flow_list += flows
            self.mask_list += masks
        sat_images = [osp.join(root, "satimages", f"{id}.png") for id in ids]
        snap_images = [osp.join(root, "snapshots", f"{id}.png") for id in ids]
        self.image_list += list(zip(sat_images, snap_images))
        self.crop = crop
        for img1 in sat_images:
            frame_id = (img1.split('/')[-1]).split('.')[0]
            self.extra_info.append(frame_id)

    def read_flow(self, index, sparse=False):
        flow = frame_utils.read_gen(self.flow_list[index])
        if sparse:
            valid = frame_utils.read_gen(self.mask_list[index])
        else:
            valid = None
        return flow, valid
    def __getitem__(self, index):
        if self.is_test:
            img1, img2, extra_info = super().__getitem__(index)
        else:
            img1, img2, flo, valid, extra_info = super().__getitem__(index)
        if self.crop:
            # apply center_crop
            img1 = center_crop(img1, self.crop)
            img2 = center_crop(img2, self.crop)
            if not self.is_test:
                flo = center_crop(flo, self.crop)
                valid = center_crop(valid, self.crop)
                # Make sure that the U and V components of the flow are finite
                valid = valid * (torch.isfinite(flo[..., 0, :, :]) & torch.isfinite(flo[...,1,:,:]))
        if self.is_test:
            return img1, img2, extra_info
        else:
            return img1, img2, flo, valid, extra_info

def fetch_dataloader(args):
    """ Create the data loader for the corresponding training set """
    if args.stage == 'asphere':
        aug_params = {'crop_size': args.image_size, 'min_scale': -0.1, 'max_scale': 1.0, 'do_flip': True}
        train_dataset = AsphereWarp(aug_params, split='training')
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=args.workers, drop_last=True)
    print('Training with %d image pairs' % len(train_dataset))
    return train_loader