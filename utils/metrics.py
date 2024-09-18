# code adapted from https://github.com/guochengqian/Magic123/blob/main/all_metrics/metric_utils.py

import torch
import torch.nn as nn
import os
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.models.optical_flow import raft_large
import matplotlib.pyplot as plt
import pickle


# import clip
from transformers import CLIPFeatureExtractor, CLIPModel, CLIPTokenizer, CLIPProcessor
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import cv2
from PIL import Image

# import torchvision.transforms as transforms
import glob
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import lpips
from os.path import join as osp



import json
from types import SimpleNamespace

inception_dims = [64, 192, 768, 2048]




def numpy_to_torch(images):
    images = images * 2.0 - 1.0
    images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()
    return images.cuda()


class LPIPSMeter:
    def __init__(
        self, net="alex", device=None, size=224
    ):  # or we can use 'alex', 'vgg' as network
        self.size = size
        self.net = net
        self.results = []
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.fn = lpips.LPIPS(net=net).eval().to(self.device)

    def measure(self):
        return np.mean(self.results)

    def report(self):
        return f"LPIPS ({self.net}) = {self.measure():.6f}"

    def read_img_list(self, img_list):
        size = self.size
        images = []

        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img[np.where(alpha == 0)] = [
                    255,
                    255,
                    255,
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            images.append(img)

        images = np.stack(images, axis=0)
        images = images.astype(np.float32) / 255.0

        return images

    # * recommend to use this function for evaluation
    @torch.no_grad()
    def score_gt(self, ref_paths, novel_paths):
        self.results = []
        # Load images
        img0, img1 = self.read_img_list(ref_paths), self.read_img_list(novel_paths)
        img0, img1 = numpy_to_torch(img0), numpy_to_torch(img1)
        img0 = F.interpolate(img0, size=(self.size, self.size), mode="area")
        img1 = F.interpolate(img1, size=(self.size, self.size), mode="area")
        if img0.shape[0] == 1:
            img0 = img0.repeat(img1.size(0), 1, 1, 1)

        self.results.append(self.fn.forward(img0, img1).cpu().squeeze().numpy())

        return self.measure(), self.results[0]


class PSNRMeter:
    def __init__(self, size=800):
        self.results = []
        self.size = size

    def read_img_list(self, img_list):
        size = self.size
        images = []
        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img[np.where(alpha == 0)] = [
                    255,
                    255,
                    255,
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            images.append(img)

        images = np.stack(images, axis=0)
        images = images.astype(np.float32) / 255.0
        return images

    def update(self, preds, truths):
        psnr_values = []
        if len(truths) == 1:
            # if only one truth image, then compare each pred to the trugh
            for img1 in preds:
                psnr = compare_psnr(
                    img1, truths.squeeze(), data_range=1.0
                )  # assuming your images are scaled to [0,1]
                psnr_values.append(psnr)
        else:
            # For each pair of images in the batches
            for img1, img2 in zip(preds, truths):
                # Compute the PSNR and add it to the list
                psnr = compare_psnr(
                    img1, img2, data_range=1.0
                )  # assuming your images are scaled to [0,1]
                psnr_values.append(psnr)

        # Convert the list of PSNR values to a numpy array
        self.results = psnr_values

    def measure(self):
        return np.mean(self.results)

    def report(self):
        return f"PSNR = {self.measure():.6f}"

    # * recommend to use this function for evaluation
    def score_gt(self, ref_paths, novel_paths):
        self.results = []
        print(f"ref_paths: {ref_paths}")
        print(f"novel_pahts: {novel_paths}")
        # [B, N, 3] or [B, H, W, 3], range[0, 1]
        preds = self.read_img_list(novel_paths)
        truths = self.read_img_list(ref_paths)
        self.update(preds, truths)
        return self.measure(), self.results


def rgb_ssim(
    img0,
    img1,
    max_val,
    filter_size=11,
    filter_sigma=1.5,
    k1=0.01,
    k2=0.03,
    return_map=False,
):
    """Evaluation metrics ssim"""
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        import scipy

        return scipy.signal.convolve2d(z, f, mode="valid")

    filt_fn = lambda z: np.stack(
        [
            convolve2d(convolve2d(z[..., i], filt[:, None]), filt[None, :])
            for i in range(z.shape[-1])
        ],
        -1,
    )
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0.0, sigma00)
    sigma11 = np.maximum(0.0, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


class SSIM:
    def __init__(self, use_gpu=True, size=512):
        super().__init__()
        self.use_gpu = use_gpu
        self.size = size

    def measure(self):
        return np.mean(self.results)

    def read_img_list(self, img_list):
        size = self.size
        images = []

        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert BGRA to BGR

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img[np.where(alpha == 0)] = [
                    255,
                    255,
                    255,
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            images.append(img)

        return images

    def update(self, preds, truths):
        ssim_values = []
        if len(truths) == 1:
            # if only one truth image, then compare each pred to the trugh
            for img1 in preds:
                val = rgb_ssim(img1, truths[0], max_val=1.0)
                ssim_values.append(val)
        else:
            for img1, img2 in zip(preds, truths):
                val = rgb_ssim(img1, img2, max_val=1.0)
                ssim_values.append(val)

        self.results = ssim_values

    @torch.no_grad()
    def score_gt(self, ref_paths, novel_paths):
        self.results = []
        # [B, N, 3] or [B, H, W, 3], range[0, 1]
        preds = self.read_img_list(novel_paths)
        truths = self.read_img_list(ref_paths)
        self.update(preds, truths)
        return self.measure(), self.results





    


