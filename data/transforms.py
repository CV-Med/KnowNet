import numpy as np
import torch
import random
from scipy.ndimage import zoom


class RandomScale3D:
    def __init__(self, scale_range=(0.85, 1.15)):
        self.scale_range = scale_range

    def __call__(self, image, target):
        scale = random.uniform(*self.scale_range)
        c, d, h, w = image.shape
        new_d, new_h, new_w = int(d * scale), int(h * scale), int(w * scale)
        if scale != 1.0:
            image_out = np.zeros_like(image)
            target_out = np.zeros_like(target)
            for i in range(c):
                scaled = zoom(image[i], scale, order=1)
                dh, hh, wh = scaled.shape
                ds, hs, ws = (d - dh) // 2, (h - hh) // 2, (w - wh) // 2
                if ds >= 0 and hs >= 0 and ws >= 0:
                    image_out[i, ds:ds + dh, hs:hs + hh, ws:ws + wh] = scaled
            scaled_t = zoom(target.astype(np.float32), scale, order=0)
            dh, hh, wh = scaled_t.shape
            ds, hs, ws = (d - dh) // 2, (h - hh) // 2, (w - wh) // 2
            if ds >= 0 and hs >= 0 and ws >= 0:
                target_out[ds:ds + dh, hs:hs + hh, ws:ws + wh] = scaled_t
            return {"image": image_out, "target": target_out}
        return {"image": image, "target": target}


class RandomGamma:
    def __init__(self, gamma_range=(0.7, 1.5)):
        self.gamma_range = gamma_range

    def __call__(self, image, target):
        gamma = random.uniform(*self.gamma_range)
        c, d, h, w = image.shape
        image_out = np.zeros_like(image)
        for i in range(c):
            ch = image[i]
            ch = ch - ch.min()
            ch_range = ch.max()
            if ch_range > 1e-6:
                ch = ch / ch_range
                ch = np.power(ch, gamma)
                ch = ch * ch_range
            image_out[i] = ch
        return {"image": image_out, "target": target}


class RandomNoise:
    def __init__(self, noise_std=0.05):
        self.noise_std = noise_std

    def __call__(self, image, target):
        noise = np.random.randn(*image.shape).astype(np.float32) * self.noise_std
        noise = noise * image.std() if image.std() > 1e-6 else noise
        return {"image": image + noise, "target": target}


class RandomContrast:
    def __init__(self, contrast_range=(0.65, 1.5)):
        self.contrast_range = contrast_range

    def __call__(self, image, target):
        factor = random.uniform(*self.contrast_range)
        c, d, h, w = image.shape
        image_out = image.copy()
        for i in range(c):
            mean = image[i].mean()
            image_out[i] = (image[i] - mean) * factor + mean
        return {"image": image_out, "target": target}


class RandomFlip:
    def __init__(self, axes=(0, 1, 2)):
        self.axes = axes

    def __call__(self, image, target):
        if random.random() < 0.5:
            axis = random.choice(self.axes)
            image = np.flip(image, axis=axis + 1).copy()
            target = np.flip(target, axis=axis).copy()
        return {"image": image, "target": target}


class PerChannelZScore:
    def __call__(self, image, target):
        c = image.shape[0]
        image_out = image.copy()
        for i in range(c):
            ch = image[i]
            mean, std = ch.mean(), ch.std()
            if std > 1e-6:
                image_out[i] = (ch - mean) / std
        return {"image": image_out, "target": target}


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        data = {"image": image, "target": target}
        for t in self.transforms:
            data = t(data["image"], data["target"])
        return data

    def __repr__(self):
        return "\n".join([str(t) for t in self.transforms])


def get_train_transforms():
    return Compose([
        RandomScale3D(scale_range=(0.85, 1.15)),
        RandomGamma(gamma_range=(0.7, 1.5)),
        RandomNoise(noise_std=0.05),
        RandomContrast(contrast_range=(0.65, 1.5)),
        RandomFlip(axes=(0, 1, 2)),
        PerChannelZScore(),
    ])


def get_val_transforms():
    return Compose([
        PerChannelZScore(),
    ])
