"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,edge,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        edge = cv2.warpPerspective(edge, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask,edge

def randomHorizontalFlip(image, mask,edge, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        edge = cv2.flip(edge, 1)

    return image, mask,edge

def randomVerticleFlip(image, mask,edge, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        edge = cv2.flip(edge, 0)

    return image, mask,edge

def randomRotate90(image, mask,edge, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)
        edge = np.rot90(edge)

    return image, mask,edge



def road_loader(id, root):
    _img = cv2.imread(os.path.join(root, '{}.tiff').format(id))
    _mask = cv2.imread(os.path.join(root + '{}.png').format(id), cv2.IMREAD_GRAYSCALE)
    _edge = cv2.imread(os.path.join(root.replace('massroad','roadedge') + '{}.png').format(id), cv2.IMREAD_GRAYSCALE)
    #
    img=cv2.resize(_img,(1024,1024))
    mask=cv2.resize(_mask,(1024,1024))
    edge = cv2.resize(_edge, (1024, 1024))
    # img=cv2.resize(_img,(512,512))
    # mask=cv2.resize(_mask,(512,512))

    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask,edge = randomShiftScaleRotate(img, mask,edge,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask,edge = randomHorizontalFlip(img, mask,edge)
    img, mask,edge= randomVerticleFlip(img, mask,edge)
    img, mask,edge = randomRotate90(img, mask,edge)

    mask = np.expand_dims(mask, axis=2)
    edge = np.expand_dims(edge, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    edge = np.array(edge, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    edge[edge >= 0.5] = 1
    edge[edge <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask,edge

class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root):
        self.ids = trainlist
        # self.loader = default_loader
        self.loader=road_loader
        self.root = root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask ,edge= self.loader(id, self.root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        edge=torch.Tensor(edge)
        return img, mask,edge

    def __len__(self):
        return len(self.ids)