import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import numpy as np
from torch.nn.utils.rnn import pad_sequence
try:
    from pytorch3d.loss import chamfer_distance
except:
    print('chamfer_distance is disabled since there is no pytorch3d distribution installed')

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel, mu=1.5):
    _1D_window = gaussian(window_size, mu).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window.requires_grad = False
    return window

def create_window_avg(window_size, channel):
    _2D_window = torch.ones(window_size, window_size).float().unsqueeze(0).unsqueeze(0) / (window_size ** 2)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window.requires_grad = False
    return window

def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map

class SSIM(torch.nn.Module):
    def __init__(self, window_size=11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel)

class AVERAGE(nn.Module):
    def __init__(self, window_size=7, size_average=False):
        super(AVERAGE, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_avg(window_size, self.channel)

    def forward(self, image):
        mu = F.avg_pool2d(image, 7, 1, self.window_size // 2, count_include_pad=False)
        return mu

class Sharpness(nn.Module):
    def __init__(self):
        super(Sharpness, self).__init__()
        self.AVG = AVERAGE()

    def gradient(self, inp):
        D_dy = inp[:, :, :, :] - F.pad(inp[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = inp[:, :, :, :] - F.pad(inp[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx, D_dy

    def sharpness(self, image):
        grad = self.gradient(image)
        mu = self.AVG(image) + 1e-8
        output = - (grad[0]**2 + grad[1]**2) - torch.abs((image - mu) / mu) - torch.pow(image - mu, 2)
        return output

    def forward(self, recon_image, image):
        rec_srp = self.sharpness(recon_image).squeeze(1)
        inp_srp = self.sharpness(image).squeeze(1)
        sharpness_loss = F.l1_loss(rec_srp, inp_srp)
        return sharpness_loss

class BlurMetric(nn.Module):
    def __init__(self, loss_fn = 'mse', device='cpu', alpha=0.8, _lambda=0.2, beta=1., sigma=1., kernel_size=7):
        super().__init__()
        if loss_fn == 'mse':
            self.loss = nn.MSELoss()
        elif loss_fn == 'recon':
            self.ssim = SSIM()
            self.l1 = nn.L1Loss()
            self.loss = lambda x, y : alpha * (1 - self.ssim(x, y)).mean()/2 + (1 - alpha) * self.l1(x, y)
        elif loss_fn == 'l1':
            self.loss = nn.L1Loss()  
        elif loss_fn == 'ssim':
            self.ssim = SSIM()
            self.loss = lambda x, y: self.ssim(x, y).mean()
        elif loss_fn == 'sharp':
            self.loss = Sharpness()
        elif loss_fn == 'smooth':
            self.loss = Smoothness(beta)
        elif loss_fn == 'blur':
            self.loss = Blur(sigma, kernel_size)
        else:
            self.mse = nn.MSELoss()
            self.loss = lambda x , y : -10. * torch.log(self.mse(x, y)) / torch.log(torch.Tensor([10.])).to(device)
    
    def forward(self, recon_image, image=None):
        return self.loss(recon_image, image)

class EvalRecon(nn.Module):
    def __init__(self, device):
        super(EvalRecon, self).__init__()
        self.psnr = BlurMetric('psnr', device)
        self.ssim = BlurMetric('ssim')
        self.mse = BlurMetric('mse')
        self.sharp = BlurMetric('sharp')

    def forward(self, recon, target):
        _psnr = self.psnr(recon, target)
        _ssim = self.ssim(recon, target)
        _mse = self.mse(recon, target)
        _sharp = self.sharp(recon, target)
        return _psnr, _ssim, _mse, _sharp

class Smoothness(nn.Module):
    def __init__(self, beta=1.):
        super(Smoothness, self).__init__()
        self.beta = beta

    def forward(self, dpt, target):
        gt_grad = self.gradient(target)
        gt_grad_x_exp = torch.exp(-gt_grad[0].abs()) * self.beta
        gt_grad_y_exp = torch.exp(-gt_grad[1].abs()) * self.beta

        dx, dy = self.gradient(dpt.unsqueeze(1))
        dD_x = dx.abs() * gt_grad_x_exp
        dD_y = dy.abs() * gt_grad_y_exp
        sm_loss = (dD_x + dD_y).mean()

        return sm_loss.unsqueeze(0)

    def gradient(self, inp):
        D_dy = inp[:, :, :, :] - F.pad(inp[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = inp[:, :, :, :] - F.pad(inp[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx, D_dy

class Blur(nn.Module):
    def __init__(self, sigma, size, beta=0.25):
        super(Blur, self).__init__()
        self.kernel = self.gen_LoG_kernel(sigma, size)
        self.beta = beta

    def forward(self, img, _):
        B, C, H, W = img.shape
        img_lap = F.conv2d(img, self.kernel.to(torch.get_device(img)), padding='same')
        blur_loss = - torch.log (torch.sum(img_lap ** 2, dim=[1, 2, 3]) / (H*W - torch.mean(img, dim=[1,2,3])**2) + 1e-8)
        return blur_loss.mean() * self.beta

    def gen_LoG_kernel(self, sigma, size):
        X = np.arange(size//2, -size//2, -1)
        Y = np.arange(size//2, -size//2, -1)
        xx, yy = np.meshgrid(X, Y)
        LoG_kernel = 1 / (np.pi * sigma ** 4) * (1 - (xx ** 2 + yy ** 2) / (2 * sigma ** 2)) * np.exp(- (xx ** 2 + yy ** 2) / (2 * sigma ** 2))    
        return torch.from_numpy(LoG_kernel).type(torch.float32).view(1, 1, size, size)
    
class BinsChamferLoss(nn.Module):
    """BinsChamferLoss used in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_. 
    
        Waiting for re-writing
        
    Args:
        loss_weight (float): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 loss_weight=1.0):
        super(BinsChamferLoss, self).__init__()
        self.loss_weight = loss_weight

    def bins_chamfer_loss(self, bins, target_depth_maps):
        bin_centers = 0.5 * (bins[:, 1:] + bins[:, :-1])
        n, p = bin_centers.shape
        input_points = bin_centers.view(n, p, 1)  # .shape = n, p, 1
        # n, c, h, w = target_depth_maps.shape

        target_points = target_depth_maps.flatten(1)  # n, hwc
        mask = target_points.ge(1e-3)  # only valid ground truth points
        target_points = [p[m] for p, m in zip(target_points, mask)]
        target_lengths = torch.Tensor([len(t) for t in target_points]).long().to(target_depth_maps.device)
        target_points = pad_sequence(target_points, batch_first=True).unsqueeze(2)  # .shape = n, T, 1

        loss, _ = chamfer_distance(x=input_points, y=target_points, y_lengths=target_lengths)
        return loss

    def forward(self,
                input,
                target):
        """Forward function."""
        
        chamfer_loss = self.bins_chamfer_loss(input, target)
        chamfer_loss = self.loss_weight * chamfer_loss
        return chamfer_loss
class SigLoss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self, args, 
                 valid_mask=True,
                 loss_weight=2.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001 # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth