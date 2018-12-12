
""" TVNet data loader """

from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import random
import cv2

def string_length_tf(t):
    def lens(s):
        return len(s)
    return tf.py_func(lens, [t], [tf.int64])

def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    lines = lines[:-1]
    f.close()
    return len(lines)

def read_file_path(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()

    f.close()
    left_path=[]
    right_path=[]
    lines = lines[:-1]
    for line in lines:
        line_split=line.split(' ')
        left_path.append(line_split[0])
        right_path.append(line_split[1].replace('\n',''))
    return left_path, right_path

class Dataloader(object):
    """TVNet data loader """

    def __init__(self, data_path, filenames_file):
        self.data_path = data_path
        self.left_image_batch = None
        self.right_image_batch = None
        self.num_samples = count_text_lines(filenames_file)
        self.left_path, self.right_path = read_file_path(filenames_file)


        if len(self.left_path) == len(self.right_path) == self.num_samples:
            print('dataloader: number of samples: {}'.format(self.num_samples))
        else:
            print('dataloader: number is not the same')



    def batch(self, batch_size):
        img1 = np.zeros((batch_size, 384, 512, 3))
        img2 = np.zeros((batch_size, 384, 512, 3))
        for i in range(batch_size):
            k = random.randint(0, self.num_samples-1)
            print(self.data_path +self.right_path[k])
            #img1_o = cv2.imread(self.data_path + self.left_path[k])
            #img2_o = cv2.imread(self.data_path + self.right_path[k])
            #img1[i, ...] = cv2.resize(img1_o , (512, 384), interpolation=cv2.INTER_LINEAR)
            #img2[i, ...] = cv2.resize(img2_o, (512, 384), interpolation=cv2.INTER_LINEAR)
            img1[i, ...] = cv2.resize(cv2.imread(self.data_path + self.left_path[k]), (512, 384), interpolation=cv2.INTER_LINEAR)
            img2[i, ...] = cv2.resize(cv2.imread(self.data_path + self.right_path[k]), (512, 384), interpolation=cv2.INTER_LINEAR)

        return img1, img2

    def get_test_image(self):
        image_1 = np.zeros((self.num_samples, 384, 512, 3))
        image_2 = np.zeros((self.num_samples, 384, 512, 3))
        for i in range(self.num_samples):
            a = cv2.imread(self.data_path + self.left_path[i])
            b = cv2.imread(self.data_path + self.right_path[i])
            if a.shape[0] != 375 & a.shape[1] != 1242:
                print(self.left_path[i])
                print(a.shape[0])
                print(a.shape[1])
            image_1[i, ...] = cv2.resize(a, (512, 384), interpolation=cv2.INTER_LINEAR)
            image_2[i, ...] = cv2.resize(b, (512, 384), interpolation=cv2.INTER_LINEAR)
        return image_1, image_2

