import os
import cv2
import glob
import multiprocessing
import numpy as np
import random
import torch
import matplotlib.pyplot as plt



class Imgloader:

    def __init__(self, data_dir='/home/spencer/PycharmProjects/CycleGAN/CycleGAN/datasets/eo_sar_256/trainA/train', num_patches=16):
        self.data_dir = data_dir
        # num_patches right now has a to be a power of 2
        self.num_patches = num_patches


    def __call__(self, filename):
        imgs = cv2.imread(self.data_dir + '/' + filename)
        # imgs = imgs.permute(2,0,1)
        return imgs


    def get_dir(self):
        return self.data_dir


class DataLoader:

    def __init__(self, batch_size=4, data_dir='/home/spencer/PycharmProjects/CycleGAN/CycleGAN/datasets/eo_sar_256/trainA/train'):
        self.BATCH_SIZE = batch_size
        self.results = []
        self.batch_num = 0
        self.epoch = 0
        self.data_dir = data_dir
        self.cpu_count = multiprocessing.cpu_count()

    def load_data(self, num_processes=multiprocessing.cpu_count(), num_sections=16):
        proc = Imgloader(self.data_dir, num_patches=num_sections)
        files = os.listdir(proc.get_dir())
        pool = multiprocessing.Pool(processes=num_processes)
        self.results = pool.map(proc, files)

    def get_batch(self):
        if self.BATCH_SIZE * self.batch_num + self.BATCH_SIZE > len(self.results):
            self.batch_num = 0
            self.epoch += 1
            random.shuffle(self.results)
        batch = self.results[self.batch_num * self.BATCH_SIZE: self.batch_num * self.BATCH_SIZE + self.BATCH_SIZE]
        self.batch_num += 1
        batch = np.array(batch)
        # plt.figure()
        # plt.imshow(output_tensor[0])
        # plt.show()

        return torch.as_tensor(batch.astype(np.float32)).permute(0,3,1,2).cuda(), self.batch_num, self.epoch


def reformat_picture(img_tensor, num_sections, patch_size):
    patches = []
    img = img_tensor.cpu().detach().numpy()
    # img = img.transpose()
    y, x = img.shape
    step_y = y // num_sections
    step_x = x // num_sections
    reconstructed_img = np.zeros((num_sections * patch_size, num_sections * patch_size))

    for i in range(num_sections ** 2):
        patches += [img[i, :].reshape((patch_size, patch_size))]
        reconstructed_img[int(i/num_sections)*patch_size:int(i/num_sections)*patch_size + patch_size, (i % num_sections)*patch_size:((i) % num_sections)*patch_size + patch_size] = patches[i]

    return reconstructed_img


if __name__ == "__main__":
    loader = DataLoader()
    loader.load_data(num_processes=24, num_sections=8)
    batch, index, epoch = loader.get_batch()
    test = reformat_picture(batch[0,:,:],8,32)
    # cv2.imshow('test', test)
    # cv2.waitKey()
    x = 1

