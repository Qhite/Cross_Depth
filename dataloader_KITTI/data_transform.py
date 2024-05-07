import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import random


import matplotlib.pyplot as plt

def _is_pil_image(img):
    return isinstance(img, Image.Image)

class Resize():
    def __init__(self, crop=True, upsample=False):
        self.crop = crop
        self.upsamle = upsample
        pass
    
    def __call__(self, data):
        image, depth = data["image"], data["depth"]

        if self.crop:
            image = self.Crop(image)
            depth = self.Crop(depth)
        image = self.CenterCrop(image)
        depth = self.CenterCrop(depth)

        if not self.upsamle:
            depth = depth.resize((568, 108), resample=0)

        return {"image": image, "depth": depth}

    def Crop(self, image):
        garg_ = [0.40810811, 0.99189189,
                 0.03594771, 0.96405229]

        w, h = image.size

        sh, eh = int(h*garg_[0]), int(h*garg_[1])
        sw, ew = int(w*garg_[2]), int(w*garg_[3])

        image = image.crop((sw, sh, ew, eh))

        return image

    def CenterCrop(self, image, size=(1136, 215)):
        w, h = image.size

        cw, ch = size

        if w == cw and h == ch:
            return image
        
        sw = int(round((w - cw) / 2.))
        sh = int(round((h - ch) / 2.))

        image = image.crop((sw, sh, sw + cw, sh + ch))

        return image

class ToTensor():
    def __init__(self):
        pass

    def __call__(self, data):
        image, depth = data["image"], data["depth"]

        image = transforms.ToTensor()(image).float()
        depth_np = np.array(depth, dtype=int)
        assert(np.max(depth) > 255)

        depth = depth_np.astype(np.float16) / 256.
        depth[depth_np == 0] = -1.

        depth = torch.tensor(depth).unsqueeze(0)
        
        return {"image": image, "depth": depth}

class Lighting(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, data):
        image, depth = data['image'], data['depth']

        if self.alphastd == 0:
            return image

        alpha = image.new(3).normal_(0, self.alphastd)

        rgb = self.eigvec.type_as(image).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        image = image.add(rgb.view(3, 1, 1).expand_as(image))

        return {'image': image, 'depth': depth}
    
class Grayscale(object):
    def __call__(self, img):
        gs = img.clone()
        gs[0].mul_(0.299).add_(0.587*gs[1]).add_(0.114*gs[2])
        gs[1].copy_(gs[0])
        gs[2].copy_(gs[0])
        return gs

class Saturation(object):
    def __init__(self, val):
        self.val = val

    def __call__(self, img):
        gs = Grayscale()(img)
        alpha = random.uniform(-self.val, self.val)
        return img.lerp(gs, alpha)

class Brightness(object):
    def __init__(self, val):
        self.val = val

    def __call__(self, img):
        gs = img.new().resize_as_(img).zero_()
        alpha = random.uniform(-self.val, self.val)

        return img.lerp(gs, alpha)

class Contrast(object):
    def __init__(self, val):
        self.val = val

    def __call__(self, img):
        gs = Grayscale()(img)
        gs.fill_(gs.mean())
        alpha = random.uniform(-self.val, self.val)
        return img.lerp(gs, alpha)

class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.transforms = []
        if brightness != 0:
            self.transforms.append(Brightness(brightness))
        if contrast != 0:
            self.transforms.append(Contrast(contrast))
        if saturation != 0:
            self.transforms.append(Saturation(saturation))
    
    def __call__(self, data):
        image, depth= data['image'], data['depth']

        if self.transforms is None:
            return {'image': image, 'depth': depth}

        for i in torch.randperm(len(self.transforms)):
            image = self.transforms[i](image)

        return {'image': image, 'depth': depth}

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        image, depth = data['image'], data['depth']

        image = transforms.Normalize(self.mean, self.std)(image)

        return {'image': image, 'depth': depth}