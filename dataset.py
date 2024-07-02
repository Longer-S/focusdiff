
from torch.utils.data import Dataset
import torch
import numpy as np
import os
from PIL import Image
import cv2
import torchvision.transforms as T

class NYUDataset(Dataset):
    def __init__(self, root_dir, split='train', shuffle=False, img_num=1, visible_img=1, focus_dist=[0.1,.15,.3,0.7,1.5], recon_all=True, 
                    RGBFD=False, DPT=False, AIF=False, scale=2, norm=True, near=0.1, far=1., trans=False, resize=256):
        self.root_dir = root_dir
        self.shuffle = shuffle
        self.img_num = img_num
        self.visible_img = visible_img
        self.focus_dist = torch.Tensor(focus_dist)
        self.recon_all = recon_all
        self.RGBFD = RGBFD
        self.DPT = DPT
        self.AIF = AIF
        self.norm = norm
        self.trans = trans
        self.near = near
        self.far = far
        if resize is not None:
            self.transform = T.Compose([
            T.Lambda(lambda t: (t * 2) - 1),
            T.Resize((resize//scale, resize//scale))
            ])
        else:
            self.transform = None

        self.aif_path = os.path.join(self.root_dir, f'{split}_rgb')
        self.dpt_path = os.path.join(self.root_dir, f'{split}_depth')
        if self.norm:
            self.all_path = os.path.join(self.root_dir, f'{split}_fs5')
        elif self.trans:
            self.all_path = os.path.join(self.root_dir, f'{split}_fs5_orig_trans')
        else:
            self.all_path = os.path.join(self.root_dir, f'{split}_fs_even')

        
        ##### Load and sort all images
        self.imglist_all = [f for f in os.listdir(self.all_path) if os.path.isfile(os.path.join(self.all_path, f))]
        self.imglist_dpt = [f for f in os.listdir(self.dpt_path) if os.path.isfile(os.path.join(self.dpt_path, f))]
        self.imglist_aif = [f for f in os.listdir(self.aif_path) if os.path.isfile(os.path.join(self.aif_path, f))]

        self.n_stack = len(self.imglist_aif)
        if split == 'train':
            print(f"{self.visible_img} out of {self.img_num} images per sample are visible for input")
        self.imglist_all.sort()
        self.imglist_dpt.sort()
        self.imglist_aif.sort()

    def __len__(self):
        return self.n_stack

    def __getitem__(self, idx):
        img_idx = idx *5

        sub_idx = np.arange(self.img_num)
        if self.shuffle:
            np.random.shuffle(sub_idx)
        input_idx = sub_idx[:self.visible_img]
        if self.recon_all:
            output_idx = sub_idx
        else:
            output_idx = sub_idx[self.visible_img:]

        mats_output = []

        for i in sub_idx:
            img_all = cv2.imread(os.path.join(self.all_path, self.imglist_all[img_idx + i]))/255.
            mat_all = torch.from_numpy(img_all.copy().astype(np.float32).transpose((2, 0, 1)))
            if self.transform is not None:
                mat_all = self.transform(mat_all)
            if i in output_idx:    
                mats_output.append(mat_all.unsqueeze(0))

        data = dict(output=torch.cat(mats_output), output_fd=self.focus_dist[output_idx])


        if self.DPT:
            img_dpt = Image.open(os.path.join(self.dpt_path, self.imglist_dpt[idx]))
            img_dpt = np.asarray(img_dpt, dtype=np.float32)/255.
            img_dpt = torch.from_numpy(img_dpt).unsqueeze(0)
            if self.transform is not None:
                img_dpt = self.transform(img_dpt)
            mat_dpt = img_dpt.repeat(3, 1, 1)
            data.update(dpt = mat_dpt)


        return data

