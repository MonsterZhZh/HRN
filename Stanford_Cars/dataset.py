import torch
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms

from os.path import join
from PIL import Image
import random
import math
import os
import networkx as nx
import numpy as np


class CarDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None, re_level='class', proportion=1.0):
        super(CarDataset, self).__init__()

        self.re_level = re_level
        self.proportion = proportion
        self.trees = [
            [1, 7],
            [2, 6],
            [3, 6],
            [4, 6],
            [5, 6],
            [6, 3],
            [7, 4],
            [8, 2],
            [9, 3],
            [10, 2],
            [11, 3],
            [12, 2],
            [13, 3],
            [14, 3],
            [15, 3],
            [16, 6],
            [17, 6],
            [18, 9],
            [19, 4],
            [20, 6],
            [21, 2],
            [22, 3],
            [23, 6],
            [24, 6],
            [25, 3],
            [26, 6],
            [27, 2],
            [28, 3],
            [29, 6],
            [30, 9],
            [31, 2],
            [32, 7],
            [33, 7],
            [34, 3],
            [35, 6],
            [36, 2],
            [37, 7],
            [38, 2],
            [39, 2],
            [40, 6],
            [41, 6],
            [42, 3],
            [43, 3],
            [44, 6],
            [45, 2],
            [46, 3],
            [47, 6],
            [48, 7],
            [49, 6],
            [50, 7],
            [51, 6],
            [52, 7],
            [53, 1],
            [54, 1],
            [55, 2],
            [56, 3],
            [57, 3],
            [58, 7],
            [59, 2],
            [60, 5],
            [61, 6],
            [62, 7],
            [63, 6],
            [64, 8],
            [65, 1],
            [66, 3],
            [67, 6],
            [68, 7],
            [69, 1],
            [70, 1],
            [71, 8],
            [72, 3],
            [73, 6],
            [74, 1],
            [75, 1],
            [76, 7],
            [77, 2],
            [78, 5],
            [79, 6],
            [80, 2],
            [81, 2],
            [82, 9],
            [83, 9],
            [84, 9],
            [85, 5],
            [86, 1],
            [87, 1],
            [88, 8],
            [89, 7],
            [90, 1],
            [91, 1],
            [92, 9],
            [93, 3],
            [94, 7],
            [95, 7],
            [96, 6],
            [97, 6],
            [98, 4],
            [99, 3],
            [100, 2],
            [101, 3],
            [102, 2],
            [103, 2],
            [104, 3],
            [105, 6],
            [106, 1],
            [107, 2],
            [108, 5],
            [109, 7],
            [110, 7],
            [111, 1],
            [112, 3],
            [113, 1],
            [114, 1],
            [115, 6],
            [116, 9],
            [117, 6],
            [118, 7],
            [119, 8],
            [120, 7],
            [121, 7],
            [122, 1],
            [123, 2],
            [124, 1],
            [125, 1],
            [126, 5],
            [127, 5],
            [128, 3],
            [129, 6],
            [130, 4],
            [131, 7],
            [132, 7],
            [133, 7],
            [134, 6],
            [135, 6],
            [136, 6],
            [137, 6],
            [138, 6],
            [139, 4],
            [140, 6],
            [141, 3],
            [142, 7],
            [143, 7],
            [144, 3],
            [145, 7],
            [146, 7],
            [147, 7],
            [148, 7],
            [149, 7],
            [150, 3],
            [151, 3],
            [152, 3],
            [153, 3],
            [154, 7],
            [155, 7],
            [156, 6],
            [157, 2],
            [158, 2],
            [159, 7],
            [160, 3],
            [161, 2],
            [162, 6],
            [163, 3],
            [164, 6],
            [165, 6],
            [166, 8],
            [167, 6],
            [168, 4],
            [169, 8],
            [170, 4],
            [171, 3],
            [172, 3],
            [173, 6],
            [174, 5],
            [175, 2],
            [176, 6],
            [177, 6],
            [178, 4],
            [179, 2],
            [180, 3],
            [181, 6],
            [182, 6],
            [183, 4],
            [184, 6],
            [185, 6],
            [186, 7],
            [187, 6],
            [188, 6],
            [189, 7],
            [190, 4],
            [191, 4],
            [192, 4],
            [193, 4],
            [194, 6],
            [195, 7],
            [196, 2]
        ]

        name_list = []
        family_label_list = []
        species_label_list = []

        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename, classname = l.strip().strip('\n').split(' ')
                name_list.append(imagename)
                class_label = int(classname)
                family_label_list.append(self.trees[class_label-1][1])
                species_label_list.append(class_label + 9)

        image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform

        self.image_filenames, self.labels = self.relabel(image_filenames, family_label_list, species_label_list)

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        target = self.labels[index] - 1

        return input, target

    def __len__(self):
        return len(self.image_filenames)

    def relabel(self, image_filenames, family_label_list, species_label_list):
        class_imgs = {}
        for i in range(len(image_filenames)):
            if str(species_label_list[i]) not in class_imgs.keys():
                class_imgs[str(species_label_list[i])] = {'images': [], 'family': []}
                class_imgs[str(species_label_list[i])]['images'].append(image_filenames[i])
                class_imgs[str(species_label_list[i])]['family'].append(family_label_list[i])
            else:
                class_imgs[str(species_label_list[i])]['images'].append(image_filenames[i])
        labels = []
        images = []
        for key in class_imgs.keys():
            # random.shuffle(class_imgs[key]['images'])
            images += class_imgs[key]['images']
            labels += [int(key)] * math.ceil(len(class_imgs[key]['images']) * self.proportion)
            rest = len(class_imgs[key]['images']) - math.ceil(len(class_imgs[key]['images']) * self.proportion)
            # print(key + ' has the rest: ' + str(rest))
            if self.re_level == 'family':
                labels += class_imgs[key]['family'] * rest
            elif self.re_level == 'class':
                labels += [int(key)] * rest
            else:
                print('Unrecognized level!!!')

        return images, labels


to_skip = [-1]
class CarDataset2(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None, re_level='class', proportion=1.0):
        super(CarDataset2, self).__init__()

        self.re_level = re_level
        self.proportion = proportion
        self.trees = [
            [1, 7],
            [2, 6],
            [3, 6],
            [4, 6],
            [5, 6],
            [6, 3],
            [7, 4],
            [8, 2],
            [9, 3],
            [10, 2],
            [11, 3],
            [12, 2],
            [13, 3],
            [14, 3],
            [15, 3],
            [16, 6],
            [17, 6],
            [18, 9],
            [19, 4],
            [20, 6],
            [21, 2],
            [22, 3],
            [23, 6],
            [24, 6],
            [25, 3],
            [26, 6],
            [27, 2],
            [28, 3],
            [29, 6],
            [30, 9],
            [31, 2],
            [32, 7],
            [33, 7],
            [34, 3],
            [35, 6],
            [36, 2],
            [37, 7],
            [38, 2],
            [39, 2],
            [40, 6],
            [41, 6],
            [42, 3],
            [43, 3],
            [44, 6],
            [45, 2],
            [46, 3],
            [47, 6],
            [48, 7],
            [49, 6],
            [50, 7],
            [51, 6],
            [52, 7],
            [53, 1],
            [54, 1],
            [55, 2],
            [56, 3],
            [57, 3],
            [58, 7],
            [59, 2],
            [60, 5],
            [61, 6],
            [62, 7],
            [63, 6],
            [64, 8],
            [65, 1],
            [66, 3],
            [67, 6],
            [68, 7],
            [69, 1],
            [70, 1],
            [71, 8],
            [72, 3],
            [73, 6],
            [74, 1],
            [75, 1],
            [76, 7],
            [77, 2],
            [78, 5],
            [79, 6],
            [80, 2],
            [81, 2],
            [82, 9],
            [83, 9],
            [84, 9],
            [85, 5],
            [86, 1],
            [87, 1],
            [88, 8],
            [89, 7],
            [90, 1],
            [91, 1],
            [92, 9],
            [93, 3],
            [94, 7],
            [95, 7],
            [96, 6],
            [97, 6],
            [98, 4],
            [99, 3],
            [100, 2],
            [101, 3],
            [102, 2],
            [103, 2],
            [104, 3],
            [105, 6],
            [106, 1],
            [107, 2],
            [108, 5],
            [109, 7],
            [110, 7],
            [111, 1],
            [112, 3],
            [113, 1],
            [114, 1],
            [115, 6],
            [116, 9],
            [117, 6],
            [118, 7],
            [119, 8],
            [120, 7],
            [121, 7],
            [122, 1],
            [123, 2],
            [124, 1],
            [125, 1],
            [126, 5],
            [127, 5],
            [128, 3],
            [129, 6],
            [130, 4],
            [131, 7],
            [132, 7],
            [133, 7],
            [134, 6],
            [135, 6],
            [136, 6],
            [137, 6],
            [138, 6],
            [139, 4],
            [140, 6],
            [141, 3],
            [142, 7],
            [143, 7],
            [144, 3],
            [145, 7],
            [146, 7],
            [147, 7],
            [148, 7],
            [149, 7],
            [150, 3],
            [151, 3],
            [152, 3],
            [153, 3],
            [154, 7],
            [155, 7],
            [156, 6],
            [157, 2],
            [158, 2],
            [159, 7],
            [160, 3],
            [161, 2],
            [162, 6],
            [163, 3],
            [164, 6],
            [165, 6],
            [166, 8],
            [167, 6],
            [168, 4],
            [169, 8],
            [170, 4],
            [171, 3],
            [172, 3],
            [173, 6],
            [174, 5],
            [175, 2],
            [176, 6],
            [177, 6],
            [178, 4],
            [179, 2],
            [180, 3],
            [181, 6],
            [182, 6],
            [183, 4],
            [184, 6],
            [185, 6],
            [186, 7],
            [187, 6],
            [188, 6],
            [189, 7],
            [190, 4],
            [191, 4],
            [192, 4],
            [193, 4],
            [194, 6],
            [195, 7],
            [196, 2]
        ]
        self.trees_order_to_family = [
            [1, 53, 54, 65, 69, 70, 74, 75, 86, 87, 90, 91, 106, 111, 113, 114, 122, 124, 125], 
            [2, 8, 10, 12, 21, 27, 31, 36, 38, 39, 45, 55, 59, 77, 80, 81, 100, 102, 103, 107, 123, 157, 158, 161, 175, 179, 196], 
            [3, 6, 9, 11, 13, 14, 15, 22, 25, 28, 34, 42, 43, 46, 56, 57, 66, 72, 93, 99, 101, 104, 112, 128, 141, 144, 150, 151, 152, 153, 160, 163, 171, 172, 180], 
            [4, 7, 19, 98, 130, 139, 168, 170, 178, 183, 190, 191, 192, 193], 
            [5, 60, 78, 85, 108, 126, 127, 174], 
            [6, 2, 3, 4, 5, 16, 17, 20, 23, 24, 26, 29, 35, 40, 41, 44, 47, 49, 51, 61, 63, 67, 73, 79, 96, 97, 105, 115, 117, 129, 134, 135, 136, 137, 138, 140, 156, 162, 164, 165, 167, 173, 176, 177, 181, 182, 184, 185, 187, 188, 194], 
            [7, 1, 32, 33, 37, 48, 50, 52, 58, 62, 68, 76, 89, 94, 95, 109, 110, 118, 120, 121, 131, 132, 133, 142, 143, 145, 146, 147, 148, 149, 154, 155, 159, 186, 189, 195], 
            [8, 64, 71, 88, 119, 166, 169], 
            [9, 18, 30, 82, 83, 84, 92, 116]
        ]
        
        self.g, self.g_t, self.adj_matrix, self.to_eval, self.nodes_idx = self.compute_adj_matrix()

        class_imgs = {}
        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename, classname = l.strip().strip('\n').split(' ')
                class_label = int(classname)
                y_ = np.zeros(206)
                y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, class_label + 9)]] = 1
                y_[self.nodes_idx[class_label + 9]] = 1
                if class_label not in class_imgs.keys():
                    class_imgs[class_label] = {'imgs': [], 'labels': []}
                class_imgs[class_label]['imgs'].append(join(image_dir, imagename))
                class_imgs[class_label]['labels'].append(y_)
        name_list = []
        label_list = []
        for key in class_imgs.keys():
            name_list += class_imgs[key]['imgs']
            label_list += class_imgs[key]['labels'][:int(math.ceil(len(class_imgs[key]['imgs']) * self.proportion))]
            rest = len(class_imgs[key]['imgs']) - math.ceil(len(class_imgs[key]['imgs']) * self.proportion)
            if self.re_level == 'class':
                continue
            elif self.re_level == 'family':
                y_ = np.zeros(206)
                family = self.trees[key-1][1]
                y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, family)]] = 1
                y_[self.nodes_idx[family]] = 1
                label_list += [y_] * int(rest)
            else:
                print('Unrecognized level!!!')

        self.input_transform = input_transform
        self.image_filenames = name_list
        self.labels = label_list


    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        input = Image.open(self.image_filenames[index]).convert('RGB')
        if self.input_transform:
            input = self.input_transform(input)
        target = self.labels[index]

        return input, target

    def __len__(self):
        return len(self.image_filenames)

    def compute_adj_matrix(self):
        g = nx.DiGraph()
        for items in self.trees_order_to_family:
            g.add_edge(items[0], -1)
            for item in items[1:]:
                g.add_edge(item + 9, items[0])
        nodes = sorted(g.nodes())
        nodes_idx = dict(zip(nodes, range(len(nodes))))
        g_t = g.reverse()
        am = nx.to_numpy_matrix(g, nodelist=nodes, order=nodes)
        to_eval = [t not in to_skip for t in nodes]
        return g, g_t, np.array(am), to_eval, nodes_idx

