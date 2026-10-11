"""
basicsr_local.py: Standalone, pure-Python compatibility module for Real-ESRGAN
Decoupled completely from deprecated external BasicSR on Python 3.13+ and PyTorch 2.x.
Provides required network architectures (RRDBNet), image degradation pipelines,
DiffJPEG, USM sharpening, model bases (SRModel, SRGANModel), and registry systems.
"""

import collections.abc
import glob
import itertools
import logging
import math
import os
import random
import sys
import time
import types
from collections import OrderedDict
from os import path as osp
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.hub import download_url_to_file, get_dir
from torch.nn import functional as F
from torch.nn import init as init

# ============================================================================ #
# 1. REGISTRY SYSTEM                                                           #
# ============================================================================ #

class Registry:
    """The registry that provides name -> object mapping for modular components."""

    def __init__(self, name):
        self._name = name
        self._obj_map = {}

    def _do_register(self, name, obj, suffix=None):
        if isinstance(suffix, str):
            name = f"{name}_{suffix}"
        self._obj_map[name] = obj

    def register(self, obj=None, suffix=None):
        if obj is None:
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class, suffix)
                return func_or_class
            return deco
        name = obj.__name__
        self._do_register(name, obj, suffix)

    def get(self, name, suffix='basicsr'):
        ret = self._obj_map.get(name)
        if ret is None and suffix:
            ret = self._obj_map.get(f"{name}_{suffix}")
        if ret is None:
            # Fallback check for case-insensitive match
            for k, v in self._obj_map.items():
                if k.lower() == name.lower():
                    return v
            raise KeyError(f"No object named '{name}' found in '{self._name}' registry!")
        return ret

    def __contains__(self, name):
        return name in self._obj_map

    def __iter__(self):
        return iter(self._obj_map.items())

    def keys(self):
        return self._obj_map.keys()


ARCH_REGISTRY = Registry('arch')
MODEL_REGISTRY = Registry('model')
DATASET_REGISTRY = Registry('dataset')
LOSS_REGISTRY = Registry('loss')
METRIC_REGISTRY = Registry('metric')

# ============================================================================ #
# 2. FILE SCANNING, LOGGING & DOWNLOAD UTILITIES                               #
# ============================================================================ #

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find files matching a suffix (generator)."""
    if suffix is not None and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        try:
            entries = os.scandir(dir_path)
        except OSError:
            return
        for entry in entries:
            if not entry.name.startswith('.') and entry.is_file():
                ret_path = entry.path if full_path else osp.relpath(entry.path, root)
                if suffix is None or ret_path.endswith(suffix):
                    yield ret_path
            elif recursive and entry.is_dir():
                yield from _scandir(entry.path, suffix=suffix, recursive=recursive)

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Download checkpoint or file from HTTP(S) URL if not already cached."""
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = file_name if file_name is not None else os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, filename))

    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=None):
    """Get root logger."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
    return logger


class FileClient:
    """Lightweight FileClient to read bytes from local disk or LMDB."""

    def __init__(self, backend='disk', **kwargs):
        self.backend = backend

    def get(self, filepath, client_key='default'):
        with open(filepath, 'rb') as f:
            return f.read()

# ============================================================================ #
# 3. IMAGE CONVERSION & PROCESSING                                             #
# ============================================================================ #

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Convert numpy image array(s) to torch tensor(s)."""
    def _totensor(img, b2r, f32):
        if img.ndim == 2:
            img = img[:, :, None]
        if img.shape[2] == 3 and b2r:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img.transpose(2, 0, 1))
        if f32:
            tensor = tensor.float()
        return tensor

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into numpy images."""
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 3:
            img_np = _tensor.permute(1, 2, 0).numpy()
            if img_np.shape[2] == 1:
                img_np = np.squeeze(img_np, axis=2)
            elif rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Unsupported tensor dimension: {n_dim}')
        if out_type == np.uint8:
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    return result[0] if len(result) == 1 else result


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes."""
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED
    }
    img = cv2.imdecode(img_np, imread_flags.get(flag, cv2.IMREAD_COLOR))
    if float32:
        img = img.astype(np.float32) / 255.0
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file."""
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise OSError(f'Failed in writing image to {file_path}.')


def filter2D(img, kernel):
    """PyTorch version of cv2.filter2D for Batched Tensors.
    img: (b, c, h, w)
    kernel: (b, k, k) or (1, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    else:
        raise ValueError(f'Kernel size must be odd, got {k}')

    ph, pw = img.size()[-2:]
    if kernel.size(0) == 1:
        img = img.reshape(b * c, 1, ph, pw)
        kernel = kernel.reshape(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).reshape(b, c, h, w)
    else:
        img = img.reshape(1, b * c, ph, pw)
        kernel = kernel.reshape(b, 1, k, k).repeat(1, c, 1, 1).reshape(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).reshape(b, c, h, w)


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening on numpy image."""
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = (np.abs(residual) * 255.0 > threshold).astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
    sharp = np.clip(img + weight * residual, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(nn.Module):
    """USM sharpening PyTorch module."""

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur
        mask = (torch.abs(residual) * 255.0 > threshold).float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = torch.clamp(img + weight * residual, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img

# ============================================================================ #
# 4. DIFFJPEG (DIFFERENTIABLE JPEG ARTIFACT SIMULATION)                         #
# ============================================================================ #

y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]).T


def diff_round(x):
    return torch.round(x) + (x - torch.round(x)) ** 3


def quality_to_factor(quality):
    if quality < 50:
        factor = 5000.0 / quality
    else:
        factor = 200.0 - quality * 2
    return factor / 100.0


class RGB2YCbCrJpeg(nn.Module):
    def __init__(self):
        super(RGB2YCbCrJpeg, self).__init__()
        matrix = np.array([[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5], [0.5, -0.418688, -0.081312]],
                          dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0.0, 128.0, 128.0]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        return result.view(image.shape)


class ChromaSubsampling(nn.Module):
    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        cb = F.avg_pool2d(image_2[:, 1:2, :, :], kernel_size=2, stride=2, count_include_pad=False)
        cr = F.avg_pool2d(image_2[:, 2:3, :, :], kernel_size=2, stride=2, count_include_pad=False)
        return image[:, :, :, 0], cb.squeeze(1), cr.squeeze(1)


class BlockSplitting(nn.Module):
    def forward(self, image):
        k = 8
        batch_size, height, width = image.shape
        image_reshaped = image.view(batch_size, height // k, k, width // k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, k, k)


class DCT8x8(nn.Module):
    def __init__(self):
        super(DCT8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1.0 / np.sqrt(2)] + [1.0] * 7)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())
        self.scale = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha) * 0.25).float())

    def forward(self, image):
        image = image - 128
        return self.scale * torch.tensordot(image, self.tensor, dims=2)


class YQuantize(nn.Module):
    def __init__(self, rounding):
        super(YQuantize, self).__init__()
        self.rounding = rounding
        self.y_table = nn.Parameter(torch.from_numpy(y_table))

    def forward(self, image, factor=1):
        if isinstance(factor, (int, float)):
            image = image.float() / (self.y_table * factor)
        else:
            b = factor.size(0)
            table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            image = image.float() / table
        return self.rounding(image)


class CQuantize(nn.Module):
    def __init__(self, rounding):
        super(CQuantize, self).__init__()
        self.rounding = rounding
        self.c_table = nn.Parameter(torch.from_numpy(c_table))

    def forward(self, image, factor=1):
        if isinstance(factor, (int, float)):
            image = image.float() / (self.c_table * factor)
        else:
            b = factor.size(0)
            table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
            image = image.float() / table
        return self.rounding(image)


class CompressJpeg(nn.Module):
    def __init__(self, rounding=torch.round):
        super(CompressJpeg, self).__init__()
        self.l1 = nn.Sequential(RGB2YCbCrJpeg(), ChromaSubsampling())
        self.l2 = nn.Sequential(BlockSplitting(), DCT8x8())
        self.c_quantize = CQuantize(rounding=rounding)
        self.y_quantize = YQuantize(rounding=rounding)

    def forward(self, image, factor=1):
        y, cb, cr = self.l1(image * 255.0)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components:
            comp = self.l2(components[k])
            comp = self.c_quantize(comp, factor=factor) if k in ('cb', 'cr') else self.y_quantize(comp, factor=factor)
            components[k] = comp
        return components['y'], components['cb'], components['cr']


class YDequantize(nn.Module):
    def __init__(self):
        super(YDequantize, self).__init__()
        self.y_table = nn.Parameter(torch.from_numpy(y_table))

    def forward(self, image, factor=1):
        if isinstance(factor, (int, float)):
            return image * (self.y_table * factor)
        b = factor.size(0)
        table = self.y_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
        return image * table


class CDequantize(nn.Module):
    def __init__(self):
        super(CDequantize, self).__init__()
        self.c_table = nn.Parameter(torch.from_numpy(c_table))

    def forward(self, image, factor=1):
        if isinstance(factor, (int, float)):
            return image * (self.c_table * factor)
        b = factor.size(0)
        table = self.c_table.expand(b, 1, 8, 8) * factor.view(b, 1, 1, 1)
        return image * table


class iDCT8x8(nn.Module):
    def __init__(self):
        super(iDCT8x8, self).__init__()
        alpha = np.array([1.0 / np.sqrt(2)] + [1.0] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float())
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float())

    def forward(self, image):
        image = image * self.alpha
        return 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128


class BlockMerging(nn.Module):
    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class ChromaUpsampling(nn.Module):
    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            h, w = x.shape[1:3]
            return x.unsqueeze(-1).repeat(1, 1, k, k).view(-1, h * k, w * k)
        cb = repeat(cb)
        cr = repeat(cr)
        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class YCbCr2RGBJpeg(nn.Module):
    def __init__(self):
        super(YCbCr2RGBJpeg, self).__init__()
        matrix = np.array([[1.0, 0.0, 1.402], [1.0, -0.344136, -0.714136], [1.0, 1.772, 0.0]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0.0, -128.0, -128.0]))
        self.matrix = nn.Parameter(torch.from_numpy(matrix))

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        return result.view(image.shape).permute(0, 3, 1, 2)


class DeCompressJpeg(nn.Module):
    def __init__(self, rounding=torch.round):
        super(DeCompressJpeg, self).__init__()
        self.c_dequantize = CDequantize()
        self.y_dequantize = YDequantize()
        self.idct = iDCT8x8()
        self.merging = BlockMerging()
        self.chroma = ChromaUpsampling()
        self.colors = YCbCr2RGBJpeg()

    def forward(self, y, cb, cr, imgh, imgw, factor=1):
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components:
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k], factor=factor)
                h, w = int(imgh / 2), int(imgw / 2)
            else:
                comp = self.y_dequantize(components[k], factor=factor)
                h, w = imgh, imgw
            comp = self.idct(comp)
            components[k] = self.merging(comp, h, w)
        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)
        image = torch.clamp(image, 0, 255)
        return image / 255.0


class DiffJPEG(nn.Module):
    """Differentiable JPEG compression simulation module."""

    def __init__(self, differentiable=True):
        super(DiffJPEG, self).__init__()
        rounding = diff_round if differentiable else torch.round
        self.compress = CompressJpeg(rounding=rounding)
        self.decompress = DeCompressJpeg(rounding=rounding)

    def forward(self, x, quality):
        factor = quality
        if isinstance(factor, (int, float)):
            factor = quality_to_factor(factor)
        else:
            factor = factor.clone()
            for i in range(factor.size(0)):
                factor[i] = quality_to_factor(factor[i].item())

        h, w = x.size()[-2:]
        h_pad = (16 - h % 16) if h % 16 != 0 else 0
        w_pad = (16 - w % 16) if w % 16 != 0 else 0
        if h_pad != 0 or w_pad != 0:
            x = F.pad(x, (0, w_pad, 0, h_pad), mode='constant', value=0)

        y, cb, cr = self.compress(x, factor=factor)
        recovered = self.decompress(y, cb, cr, (h + h_pad), (w + w_pad), factor=factor)
        return recovered[:, :, 0:h, 0:w]

# ============================================================================ #
# 5. NOISE & DEGRADATION PIPELINES (PyTorch Native)                            #
# ============================================================================ #

def _rgb_to_grayscale_pt(img):
    """Pure PyTorch RGB to grayscale converter (avoids torchvision operator dependencies)."""
    weights = img.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (img * weights).sum(dim=1, keepdim=True)


def generate_gaussian_noise_pt(img, sigma=10, gray_noise=0):
    b, c, h, w = img.size()
    if not isinstance(sigma, (float, int)):
        sigma = sigma.view(b, 1, 1, 1)
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        noise_gray = torch.randn(*[b, 1, h, w], dtype=img.dtype, device=img.device) * sigma / 255.0
        noise_gray = noise_gray.repeat(1, c, 1, 1)
    noise = torch.randn(*img.size(), dtype=img.dtype, device=img.device) * sigma / 255.0
    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    return noise


def random_generate_gaussian_noise_pt(img, sigma_range=(0, 10), gray_prob=0):
    sigma = torch.rand(img.size(0), dtype=img.dtype, device=img.device) * (sigma_range[1] - sigma_range[0]) + sigma_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_gaussian_noise_pt(img, sigma, gray_noise)


def random_add_gaussian_noise_pt(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_gaussian_noise_pt(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0
    return out


def generate_poisson_noise_pt(img, scale=1.0, gray_noise=0):
    b, _, h, w = img.size()
    if isinstance(gray_noise, (float, int)):
        cal_gray_noise = gray_noise > 0
    else:
        gray_noise = gray_noise.view(b, 1, 1, 1)
        cal_gray_noise = torch.sum(gray_noise) > 0

    if cal_gray_noise:
        img_gray = _rgb_to_grayscale_pt(img)
        img_gray = torch.clamp((img_gray * 255.0).round(), 0, 255) / 255.0
        vals_list = [len(torch.unique(img_gray[i, :, :, :])) for i in range(b)]
        vals_list = [2 ** np.ceil(np.log2(max(v, 1))) for v in vals_list]
        vals = img_gray.new_tensor(vals_list).view(b, 1, 1, 1)
        out = torch.poisson(img_gray * vals) / vals
        noise_gray = (out - img_gray).repeat(1, 3, 1, 1)

    img_clamped = torch.clamp((img * 255.0).round(), 0, 255) / 255.0
    vals_list = [len(torch.unique(img_clamped[i, :, :, :])) for i in range(b)]
    vals_list = [2 ** np.ceil(np.log2(max(v, 1))) for v in vals_list]
    vals = img_clamped.new_tensor(vals_list).view(b, 1, 1, 1)
    out = torch.poisson(img_clamped * vals) / vals
    noise = out - img_clamped

    if cal_gray_noise:
        noise = noise * (1 - gray_noise) + noise_gray * gray_noise
    if not isinstance(scale, (float, int)):
        scale = scale.view(b, 1, 1, 1)
    return noise * scale


def random_generate_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0):
    scale = torch.rand(img.size(0), dtype=img.dtype, device=img.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    gray_noise = torch.rand(img.size(0), dtype=img.dtype, device=img.device)
    gray_noise = (gray_noise < gray_prob).float()
    return generate_poisson_noise_pt(img, scale, gray_noise)


def random_add_poisson_noise_pt(img, scale_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_poisson_noise_pt(img, scale_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = torch.clamp((out * 255.0).round(), 0, 255) / 255.0
    elif clip:
        out = torch.clamp(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.0
    return out


def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """2D circular low-pass sinc filter."""
    assert kernel_size % 2 == 1, 'Kernel size must be odd.'
    try:
        from scipy import special
        kernel = np.fromfunction(
            lambda x, y: cutoff * special.j1(cutoff * np.sqrt(
                (x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)) / (
                2 * np.pi * np.sqrt((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2)),
            [kernel_size, kernel_size]
        )
    except Exception:
        d = kernel_size // 2
        y, x = np.ogrid[-d:d + 1, -d:d + 1]
        kernel = np.exp(-(x * x + y * y) / (2.0 * max((cutoff * d), 0.1) ** 2))

    kernel[(kernel_size - 1) // 2, (kernel_size - 1) // 2] = cutoff ** 2 / (4 * np.pi)
    kernel = kernel / np.sum(kernel)
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    return kernel


def random_mixed_kernels(kernel_list, kernel_prob, kernel_size=21, **kwargs):
    """Randomly generate mixed kernels."""
    d = kernel_size // 2
    y, x = np.ogrid[-d:d + 1, -d:d + 1]
    sigma = random.uniform(0.6, 5.0)
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)

# ============================================================================ #
# 6. DATA TRANSFORMS & DATA UTILITIES                                          #
# ============================================================================ #

def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale, gt_path=None):
    """Paired random crop supporting both Tensors and Numpy arrays."""
    if not isinstance(img_gts, list):
        img_gts = [img_gts]
    if not isinstance(img_lqs, list):
        img_lqs = [img_lqs]

    is_tensor = torch.is_tensor(img_gts[0])
    if is_tensor:
        h_lq, w_lq = img_lqs[0].size()[-2:]
        h_gt, w_gt = img_gts[0].size()[-2:]
    else:
        h_lq, w_lq = img_lqs[0].shape[0:2]
        h_gt, w_gt = img_gts[0].shape[0:2]

    lq_patch_size = gt_patch_size // scale
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatch: GT ({h_gt}, {w_gt}) is not {scale}x of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size ({lq_patch_size}).')

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    if is_tensor:
        img_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_lqs]
    else:
        img_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_lqs]

    top_gt, left_gt = int(top * scale), int(left * scale)
    if is_tensor:
        img_gts = [v[:, :, top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size] for v in img_gts]
    else:
        img_gts = [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in img_gts]

    ret_gt = img_gts[0] if len(img_gts) == 1 else img_gts
    ret_lq = img_lqs[0] if len(img_lqs) == 1 else img_lqs
    return ret_gt, ret_lq


def augment(imgs, hflip=True, rotation=True, flows=None, return_status=False):
    """Horizontal flip and rotation augmentations for list of numpy images."""
    hflip_status = hflip and random.random() < 0.5
    vflip_status = rotation and random.random() < 0.5
    rot90_status = rotation and random.random() < 0.5

    def _augment(img):
        if hflip_status:
            img = cv2.flip(img, 1)
        if vflip_status:
            img = cv2.flip(img, 0)
        if rot90_status:
            img = img.transpose(1, 0, 2)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    ret_imgs = imgs[0] if len(imgs) == 1 else imgs
    if return_status:
        return ret_imgs, (hflip_status, vflip_status, rot90_status)
    return ret_imgs


def paired_paths_from_folder(folders, keys, filename_tmpl='{}'):
    assert len(folders) == 2, 'Must provide exactly 2 folders for paired paths.'
    gt_paths = sorted([os.path.join(folders[0], f) for f in os.listdir(folders[0]) if os.path.isfile(os.path.join(folders[0], f))])
    lq_paths = sorted([os.path.join(folders[1], f) for f in os.listdir(folders[1]) if os.path.isfile(os.path.join(folders[1], f))])
    return [{keys[0]: gt, keys[1]: lq} for gt, lq in zip(gt_paths, lq_paths)]


def paired_paths_from_lmdb(folders, keys):
    return []


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation."""
    if not inplace:
        tensor = tensor.clone()
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return tensor.sub_(mean).div_(std)

# ============================================================================ #
# 7. RRDBNET ARCHITECTURE (STRICT WEIGHTS COMPATIBILITY GUARANTEED)             #
# ============================================================================ #

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    return nn.Sequential(*[basic_block(**kwarg) for _ in range(num_basic_block)])


def pixel_unshuffle(x, scale):
    """Pixel unshuffle operation using native PyTorch implementation."""
    return F.pixel_unshuffle(x, downscale_factor=scale)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block (RDB) for ESRGAN / Real-ESRGAN."""

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (RRDB) for ESRGAN / Real-ESRGAN."""

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    """Residual in Residual Dense Block Network (RRDBNet).
    Preserves exact layer structure and state_dict keys for official pretrained weights.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling layers
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # Upsample blocks
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

# ============================================================================ #
# 8. BASE MODEL & SR MODEL BASE CLASSES                                        #
# ============================================================================ #

class BaseModel:
    """Base model class with training and checkpointing utilities."""

    def __init__(self, opt):
        self.opt = opt
        num_gpu = opt.get('num_gpu', 1 if torch.cuda.is_available() else 0)
        self.device = torch.device('cuda' if (num_gpu != 0 and torch.cuda.is_available()) else 'cpu')
        self.is_train = opt.get('is_train', False)
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self, current_iter):
        pass

    def get_current_visuals(self):
        return {}

    def save(self, epoch, current_iter):
        pass

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        pass

    def model_ema(self, decay=0.999):
        if not hasattr(self, 'net_g_ema') or self.net_g_ema is None:
            return
        net_g = self.get_bare_model(self.net_g)
        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())
        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def get_current_log(self):
        return getattr(self, 'log_dict', {})

    def model_to_device(self, net):
        return net.to(self.device)

    def get_bare_model(self, net):
        if isinstance(net, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return net.module
        return net

    def print_network(self, net):
        pass

    def load_network(self, net, load_path, strict=True, param_key='params'):
        net = self.get_bare_model(net)
        try:
            loadnet = torch.load(load_path, map_location=self.device, weights_only=False)
        except TypeError:
            loadnet = torch.load(load_path, map_location=self.device)

        if param_key in loadnet:
            loadnet = loadnet[param_key]
        elif 'params_ema' in loadnet:
            loadnet = loadnet['params_ema']
        net.load_state_dict(loadnet, strict=strict)

    def reduce_loss_dict(self, loss_dict):
        log_dict = OrderedDict()
        for name, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                log_dict[name] = value.mean().item()
            else:
                log_dict[name] = value
        return log_dict


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """SR Model base class for Super-Resolution networks."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.net_g = None

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        pass


@MODEL_REGISTRY.register()
class SRGANModel(SRModel):
    """SRGAN Model base class with generator and discriminator optimizers."""

    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        self.net_d = None
        self.optimizer_g = None
        self.optimizer_d = None
        self.cri_pix = None
        self.cri_perceptual = None
        self.cri_gan = None
        self.net_d_iters = opt.get('train', {}).get('net_d_iters', 1)
        self.net_d_init_iters = opt.get('train', {}).get('net_d_init_iters', 0)
        self.ema_decay = opt.get('train', {}).get('ema_decay', 0)

# ============================================================================ #
# 9. DYNAMIC SYS.MODULES INJECTION                                             #
# Ensures any legacy import `from basicsr...` resolves to basicsr_local!        #
# ============================================================================ #

def _install_basicsr_shims():
    """Inject this module into sys.modules as `basicsr` and its subpackages."""
    this_mod = sys.modules[__name__]

    shims = {
        'basicsr': this_mod,
        'basicsr.archs': types.ModuleType('basicsr.archs'),
        'basicsr.archs.rrdbnet_arch': types.ModuleType('basicsr.archs.rrdbnet_arch'),
        'basicsr.archs.srvgg_arch': types.ModuleType('basicsr.archs.srvgg_arch'),
        'basicsr.data': types.ModuleType('basicsr.data'),
        'basicsr.data.data_util': types.ModuleType('basicsr.data.data_util'),
        'basicsr.data.degradations': types.ModuleType('basicsr.data.degradations'),
        'basicsr.data.transforms': types.ModuleType('basicsr.data.transforms'),
        'basicsr.models': types.ModuleType('basicsr.models'),
        'basicsr.models.base_model': types.ModuleType('basicsr.models.base_model'),
        'basicsr.models.sr_model': types.ModuleType('basicsr.models.sr_model'),
        'basicsr.models.srgan_model': types.ModuleType('basicsr.models.srgan_model'),
        'basicsr.utils': types.ModuleType('basicsr.utils'),
        'basicsr.utils.download_util': types.ModuleType('basicsr.utils.download_util'),
        'basicsr.utils.registry': types.ModuleType('basicsr.utils.registry'),
        'basicsr.utils.img_process_util': types.ModuleType('basicsr.utils.img_process_util'),
        'basicsr.utils.img_util': types.ModuleType('basicsr.utils.img_util'),
        'basicsr.utils.diffjpeg': types.ModuleType('basicsr.utils.diffjpeg'),
        'basicsr.utils.misc': types.ModuleType('basicsr.utils.misc'),
    }

    # Populate shims
    shims['basicsr.archs.rrdbnet_arch'].RRDBNet = RRDBNet
    shims['basicsr.archs.rrdbnet_arch'].RRDB = RRDB
    shims['basicsr.archs.rrdbnet_arch'].ResidualDenseBlock = ResidualDenseBlock
    shims['basicsr.archs.rrdbnet_arch'].ARCH_REGISTRY = ARCH_REGISTRY

    shims['basicsr.utils.download_util'].load_file_from_url = load_file_from_url
    shims['basicsr.utils.registry'].Registry = Registry
    shims['basicsr.utils.registry'].ARCH_REGISTRY = ARCH_REGISTRY
    shims['basicsr.utils.registry'].MODEL_REGISTRY = MODEL_REGISTRY
    shims['basicsr.utils.registry'].DATASET_REGISTRY = DATASET_REGISTRY
    shims['basicsr.utils.registry'].LOSS_REGISTRY = LOSS_REGISTRY
    shims['basicsr.utils.registry'].METRIC_REGISTRY = METRIC_REGISTRY

    shims['basicsr.utils'].scandir = scandir
    shims['basicsr.utils'].DiffJPEG = DiffJPEG
    shims['basicsr.utils'].USMSharp = USMSharp
    shims['basicsr.utils'].filter2D = filter2D
    shims['basicsr.utils'].img2tensor = img2tensor
    shims['basicsr.utils'].tensor2img = tensor2img
    shims['basicsr.utils'].imfrombytes = imfrombytes
    shims['basicsr.utils'].imwrite = imwrite
    shims['basicsr.utils'].load_file_from_url = load_file_from_url
    shims['basicsr.utils'].FileClient = FileClient
    shims['basicsr.utils'].get_root_logger = get_root_logger

    shims['basicsr.utils.img_process_util'].filter2D = filter2D
    shims['basicsr.utils.img_process_util'].USMSharp = USMSharp
    shims['basicsr.utils.img_process_util'].usm_sharp = usm_sharp

    shims['basicsr.utils.diffjpeg'].DiffJPEG = DiffJPEG
    shims['basicsr.utils.misc'].scandir = scandir

    shims['basicsr.data.degradations'].random_add_gaussian_noise_pt = random_add_gaussian_noise_pt
    shims['basicsr.data.degradations'].random_add_poisson_noise_pt = random_add_poisson_noise_pt
    shims['basicsr.data.degradations'].circular_lowpass_kernel = circular_lowpass_kernel
    shims['basicsr.data.degradations'].random_mixed_kernels = random_mixed_kernels

    shims['basicsr.data.transforms'].paired_random_crop = paired_random_crop
    shims['basicsr.data.transforms'].augment = augment

    shims['basicsr.data.data_util'].paired_paths_from_folder = paired_paths_from_folder
    shims['basicsr.data.data_util'].paired_paths_from_lmdb = paired_paths_from_lmdb

    shims['basicsr.models.base_model'].BaseModel = BaseModel
    shims['basicsr.models.sr_model'].SRModel = SRModel
    shims['basicsr.models.srgan_model'].SRGANModel = SRGANModel

    for name, module in shims.items():
        if name not in sys.modules:
            sys.modules[name] = module

_install_basicsr_shims()
