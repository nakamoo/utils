# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
import random
try:
   import cPickle as pickle
except:
   import pickle


def load_image_list(path):
    tuples = []
    with open(path, 'r') as f:
        for line in f:
            pair = line.strip().split()
            tuples.append(pair)
    return tuples


def random_brightness(image, max_delta=63, seed=None):
    delta = np.random.uniform(-max_delta, max_delta)
    newimg = image + delta
    return newimg


def random_contrast(image, lower, upper, seed=None):
    f = np.random.uniform(-lower, upper)
    mean = (image[0] + image[1] + image[2]).astype(np.float32) / 3
    ximg = np.zeros(image.shape, np.float32)
    for i in range(0, 3):
        ximg[i] = (image[i] - mean) * f + mean
    return ximg


def image_whitening(img):
    img = img.astype(np.float32)
    d, w, h = img.shape
    num_pixels = d * w * h
    mean = img.mean()
    variance = np.mean(np.square(img)) - np.square(mean)
    stddev = np.sqrt(variance)
    min_stddev = 1.0 / np.sqrt(num_pixels)
    scale = stddev if stddev > min_stddev else min_stddev
    img -= mean
    img /= scale
    return img


class Data(object):
    def __init__(self):
        data = load_image_list('data.txt')
        random.shuffle(data)
        n_data = len(data)
        self.insize = 128
        self.train = data[:n_data / 5 * -1]
        self.test = data[n_data / 5 * -1:]
        self.N = len(self.train)
        self.crop_noize = 7
        self.TEST_N = len(self.test)

    def read_image(self, path, flip=True):
        # Data loading routine
        resize_img = Image.open(path).resize((self.insize + self.crop_noize
                                              , self.insize + self.crop_noize))
        img = np.asarray(resize_img).transpose(2, 0, 1)

        # random crop
        top = random.randint(0, self.crop_noize)
        left = random.randint(0, self.crop_noize)
        bottom = self.insize + top
        right = self.insize + left
        image = img[:, top:bottom, left:right].astype(np.float32)

        # left-right flipping
        if flip and random.randint(0, 1) == 0:
            image = image[:, :, ::-1]
        # random brightness
        if random.randint(0, 4) != 0:
            image = random_brightness(image)
        # random contrast
        if random.randint(0, 4) != 0:
            image = random_contrast(image, lower=0.2, upper=1.8)
        # whitening
        image = image_whitening(image)

        image /= 255
        return image

    def get(self, index, test=False):
        # send list or tuple data (not numpy)
        x_batch = np.ndarray(
            (len(index), 3, self.insize, self.insize), dtype=np.float32)
        t_batch = np.ndarray(
            (len(index),), dtype=np.int32)

        if test:
            data_set = self.test
        else:
            data_set = self.train

        for k, idx in enumerate(index):
            path = data_set[idx][0]
            target = data_set[idx][1]
            img = self.read_image(path)
            #TODO: tolist may be very slow!
            x_batch[k] = img
            t_batch[k] = target

        return x_batch, t_batch
