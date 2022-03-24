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


class CubDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None, re_level='class', proportion=1.0):
        super(CubDataset, self).__init__()

        self.re_level = re_level
        self.proportion = proportion
        self.trees = [
            [1,12,35],
            [2,12,35],
            [3,12,35],
            [4,6,9],
            [5,4,4],
            [6,4,4],
            [7,4,4],
            [8,4,4],
            [9,8,18],
            [10,8,18],
            [11,8,18],
            [12,8,18],
            [13,8,18],
            [14,8,13],
            [15,8,13],
            [16,8,13],
            [17,8,13],
            [18,8,26],
            [19,8,21],
            [20,8,19],
            [21,8,24],
            [22,3,3],
            [23,13,37],
            [24,13,37],
            [25,13,37],
            [26,8,18],
            [27,8,18],
            [28,8,14],
            [29,8,15],
            [30,8,15],
            [31,6,9],
            [32,6,9],
            [33,6,9],
            [34,8,16],
            [35,8,16],
            [36,10,33],
            [37,8,30],
            [38,8,30],
            [39,8,30],
            [40,8,30],
            [41,8,30],
            [42,8,30],
            [43,8,30],
            [44,13,38],
            [45,12,36],
            [46,1,1],
            [47,8,16],
            [48,8,16],
            [49,8,18],
            [50,11,34],
            [51,11,34],
            [52,11,34],
            [53,11,34],
            [54,8,13],
            [55,8,16],
            [56,8,16],
            [57,8,13],
            [58,4,4],
            [59,4,5],
            [60,4,5],
            [61,4,5],
            [62,4,5],
            [63,4,5],
            [64,4,5],
            [65,4,5],
            [66,4,5],
            [67,2,2],
            [68,2,2],
            [69,2,2],
            [70,2,2],
            [71,4,6],
            [72,4,6],
            [73,8,15],
            [74,8,15],
            [75,8,15],
            [76,8,24],
            [77,8,30],
            [78,8,30],
            [79,5,7],
            [80,5,7],
            [81,5,7],
            [82,5,7],
            [83,5,7],
            [84,5,8],
            [85,8,11],
            [86,7,10],
            [87,1,1],
            [88,8,18],
            [89,1,1],
            [90,1,1],
            [91,8,21],
            [92,3,3],
            [93,8,15],
            [94,8,27],
            [95,8,18],
            [96,8,18],
            [97,8,18],
            [98,8,18],
            [99,8,23],
            [100,9,32],
            [101,9,32],
            [102,8,30],
            [103,8,30],
            [104,8,22],
            [105,3,3],
            [106,4,4],
            [107,8,15],
            [108,8,15],
            [109,8,23],
            [110,6,9],
            [111,8,20],
            [112,8,20],
            [113,8,24],
            [114,8,24],
            [115,8,24],
            [116,8,24],
            [117,8,24],
            [118,8,25],
            [119,8,24],
            [120,8,24],
            [121,8,24],
            [122,8,24],
            [123,8,24],
            [124,8,24],
            [125,8,24],
            [126,8,24],
            [127,8,24],
            [128,8,24],
            [129,8,24],
            [130,8,24],
            [131,8,24],
            [132,8,24],
            [133,8,24],
            [134,8,28],
            [135,8,17],
            [136,8,17],
            [137,8,17],
            [138,8,17],
            [139,8,13],
            [140,8,13],
            [141,4,5],
            [142,4,5],
            [143,4,5],
            [144,4,5],
            [145,4,5],
            [146,4,5],
            [147,4,5],
            [148,8,24],
            [149,8,21],
            [150,8,21],
            [151,8,31],
            [152,8,31],
            [153,8,31],
            [154,8,31],
            [155,8,31],
            [156,8,31],
            [157,8,31],
            [158,8,23],
            [159,8,23],
            [160,8,23],
            [161,8,23],
            [162,8,23],
            [163,8,23],
            [164,8,23],
            [165,8,23],
            [166,8,23],
            [167,8,23],
            [168,8,23],
            [169,8,23],
            [170,8,23],
            [171,8,23],
            [172,8,23],
            [173,8,23],
            [174,8,23],
            [175,8,23],
            [176,8,23],
            [177,8,23],
            [178,8,23],
            [179,8,23],
            [180,8,23],
            [181,8,23],
            [182,8,23],
            [183,8,23],
            [184,8,23],
            [185,8,12],
            [186,8,12],
            [187,10,33],
            [188,10,33],
            [189,10,33],
            [190,10,33],
            [191,10,33],
            [192,10,33],
            [193,8,29],
            [194,8,29],
            [195,8,29],
            [196,8,29],
            [197,8,29],
            [198,8,29],
            [199,8,29],
            [200,8,23]
        ]

        name_list = []
        family_label_list = []
        species_label_list = []

        with open(list_path, 'r') as f:
            for l in f.readlines():
                imagename, class_label, genus_label, family_label, order_label = l.strip().split(' ')
                name_list.append(imagename)
                family_label_list.append(self.trees[int(class_label)-1][-1] + 13)
                species_label_list.append(int(class_label) + 51)

        image_filenames = [join(image_dir, x) for x in name_list]
        self.input_transform = input_transform

        self.image_filenames, self.labels = self.relabel(image_filenames, family_label_list, species_label_list)

    def __getitem__(self, index):
        imagename = self.image_filenames[index]
        target = self.labels[index] - 1
        input = Image.open(self.image_filenames[index]).convert('RGB')
        
        # if target < 51:
        #     # DownSampling for samples labeled as coarse-grained
        #     loader = transforms.Compose([transforms.ToTensor()])
        #     unloader = transforms.ToPILImage()
        #     input = loader(input).unsqueeze(0)
        #     input = F.interpolate(input, scale_factor=0.25, recompute_scale_factor=True)
        #     input = input.squeeze(0)
        #     input = unloader(input)
        
        if self.input_transform:
            input = self.input_transform(input)
        
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
            images += class_imgs[key]['images']
            labels += [int(key)] * math.ceil(len(class_imgs[key]['images']) * self.proportion)
            rest = len(class_imgs[key]['images']) - math.ceil(len(class_imgs[key]['images']) * self.proportion)
            if self.re_level == 'family':
                labels += class_imgs[key]['family'] * rest
            elif self.re_level == 'class':
                labels += [int(key)] * rest
            else:
                print('Unrecognized level!!!')

        return images, labels



to_skip = [-1]
class CubDataset2(data.Dataset):
    def __init__(self, image_dir, input_transform=None, re_level='class', proportion=1.0):
        super(CubDataset2, self).__init__()

        self.re_level = re_level
        self.proportion = proportion
        self.trees = [
            [1,12,35],
            [2,12,35],
            [3,12,35],
            [4,6,9],
            [5,4,4],
            [6,4,4],
            [7,4,4],
            [8,4,4],
            [9,8,18],
            [10,8,18],
            [11,8,18],
            [12,8,18],
            [13,8,18],
            [14,8,13],
            [15,8,13],
            [16,8,13],
            [17,8,13],
            [18,8,26],
            [19,8,21],
            [20,8,19],
            [21,8,24],
            [22,3,3],
            [23,13,37],
            [24,13,37],
            [25,13,37],
            [26,8,18],
            [27,8,18],
            [28,8,14],
            [29,8,15],
            [30,8,15],
            [31,6,9],
            [32,6,9],
            [33,6,9],
            [34,8,16],
            [35,8,16],
            [36,10,33],
            [37,8,30],
            [38,8,30],
            [39,8,30],
            [40,8,30],
            [41,8,30],
            [42,8,30],
            [43,8,30],
            [44,13,38],
            [45,12,36],
            [46,1,1],
            [47,8,16],
            [48,8,16],
            [49,8,18],
            [50,11,34],
            [51,11,34],
            [52,11,34],
            [53,11,34],
            [54,8,13],
            [55,8,16],
            [56,8,16],
            [57,8,13],
            [58,4,4],
            [59,4,5],
            [60,4,5],
            [61,4,5],
            [62,4,5],
            [63,4,5],
            [64,4,5],
            [65,4,5],
            [66,4,5],
            [67,2,2],
            [68,2,2],
            [69,2,2],
            [70,2,2],
            [71,4,6],
            [72,4,6],
            [73,8,15],
            [74,8,15],
            [75,8,15],
            [76,8,24],
            [77,8,30],
            [78,8,30],
            [79,5,7],
            [80,5,7],
            [81,5,7],
            [82,5,7],
            [83,5,7],
            [84,5,8],
            [85,8,11],
            [86,7,10],
            [87,1,1],
            [88,8,18],
            [89,1,1],
            [90,1,1],
            [91,8,21],
            [92,3,3],
            [93,8,15],
            [94,8,27],
            [95,8,18],
            [96,8,18],
            [97,8,18],
            [98,8,18],
            [99,8,23],
            [100,9,32],
            [101,9,32],
            [102,8,30],
            [103,8,30],
            [104,8,22],
            [105,3,3],
            [106,4,4],
            [107,8,15],
            [108,8,15],
            [109,8,23],
            [110,6,9],
            [111,8,20],
            [112,8,20],
            [113,8,24],
            [114,8,24],
            [115,8,24],
            [116,8,24],
            [117,8,24],
            [118,8,25],
            [119,8,24],
            [120,8,24],
            [121,8,24],
            [122,8,24],
            [123,8,24],
            [124,8,24],
            [125,8,24],
            [126,8,24],
            [127,8,24],
            [128,8,24],
            [129,8,24],
            [130,8,24],
            [131,8,24],
            [132,8,24],
            [133,8,24],
            [134,8,28],
            [135,8,17],
            [136,8,17],
            [137,8,17],
            [138,8,17],
            [139,8,13],
            [140,8,13],
            [141,4,5],
            [142,4,5],
            [143,4,5],
            [144,4,5],
            [145,4,5],
            [146,4,5],
            [147,4,5],
            [148,8,24],
            [149,8,21],
            [150,8,21],
            [151,8,31],
            [152,8,31],
            [153,8,31],
            [154,8,31],
            [155,8,31],
            [156,8,31],
            [157,8,31],
            [158,8,23],
            [159,8,23],
            [160,8,23],
            [161,8,23],
            [162,8,23],
            [163,8,23],
            [164,8,23],
            [165,8,23],
            [166,8,23],
            [167,8,23],
            [168,8,23],
            [169,8,23],
            [170,8,23],
            [171,8,23],
            [172,8,23],
            [173,8,23],
            [174,8,23],
            [175,8,23],
            [176,8,23],
            [177,8,23],
            [178,8,23],
            [179,8,23],
            [180,8,23],
            [181,8,23],
            [182,8,23],
            [183,8,23],
            [184,8,23],
            [185,8,12],
            [186,8,12],
            [187,10,33],
            [188,10,33],
            [189,10,33],
            [190,10,33],
            [191,10,33],
            [192,10,33],
            [193,8,29],
            [194,8,29],
            [195,8,29],
            [196,8,29],
            [197,8,29],
            [198,8,29],
            [199,8,29],
            [200,8,23]
        ]
        self.trees_order_to_family = [
            [1, 1],
            [2, 2],
            [3, 3],
            [4, 4, 5, 6],
            [5, 7, 8],
            [6, 9],
            [7, 10],
            [8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
            [9, 32],
            [10, 33],
            [11, 34],
            [12, 35, 36],
            [13, 37, 38],
        ]
        self.trees_family_to_species = [
            [1, 46, 87, 89, 90],
            [2, 67, 68, 69, 70],
            [3, 22, 92, 105],
            [4, 5, 6, 7, 8, 58, 106],
            [5, 59, 60, 61, 62, 63, 64, 65, 66, 141, 142, 143, 144, 145, 146, 147],
            [6, 71, 72],
            [7, 79, 80, 81, 82, 83],
            [8, 84],
            [9, 4, 31, 32, 33, 110],
            [10, 86],
            [11, 85],
            [12, 185, 186],
            [13, 14, 15, 16, 17, 54, 57, 139, 140],
            [14, 28],
            [15, 29, 30, 73, 74, 75, 93, 107, 108],
            [16, 34, 35, 47, 48, 55, 56],
            [17, 135, 136, 137, 138],
            [18, 9, 10, 11, 12, 13, 26, 27, 49, 88, 95, 96, 97, 98],
            [19, 20],
            [20, 111, 112],
            [21, 19, 91, 149, 150],
            [22, 104],
            [23, 99, 109, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 200],
            [24, 21, 76, 113, 114, 115, 116, 117, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 148],
            [25, 118],
            [26, 18],
            [27, 94],
            [28, 134],
            [29, 193, 194, 195, 196, 197, 198, 199],
            [30, 37, 38, 39, 40, 41, 42, 43, 77, 78, 102, 103],
            [31, 151, 152, 153, 154, 155, 156, 157],
            [32, 100, 101],
            [33, 36, 187, 188, 189, 190, 191, 192],
            [34, 50, 51, 52, 53],
            [35, 1, 2, 3],
            [36, 45],
            [37, 23, 24, 25],
            [38, 44],
        ]

        self.g, self.g_t, self.adj_matrix, self.to_eval, self.nodes_idx = self.compute_adj_matrix()

        name_list = []
        label_list = []
        classes = os.listdir(image_dir)
        for cls in classes:
            tmp_name_list = []
            tmp_class_label_list = []
            cls_imgs = join(image_dir, cls)
            imgs = os.listdir(cls_imgs)
            y_ = np.zeros(252)
            cls_name = cls.strip().split('_')[-1]
            y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, int(cls_name) + 51)]] = 1
            y_[self.nodes_idx[int(cls_name) + 51]] = 1
            for img in imgs:
                tmp_name_list.append(join(image_dir, cls, img))
                tmp_class_label_list.append(y_)

            name_list += tmp_name_list
            label_list += tmp_class_label_list[:int(math.ceil(len(tmp_class_label_list) * self.proportion))]
            rest = len(tmp_class_label_list) - math.ceil(len(tmp_class_label_list) * self.proportion)
            y_ = np.zeros(252)
            if self.re_level == 'class':
                continue
            elif self.re_level == 'order':
                order = self.trees[int(cls_name)-1][1]
                y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, order)]] = 1
                y_[self.nodes_idx[order]] = 1
                label_list += [y_] * int(rest)
            elif self.re_level == 'family':
                family = self.trees[int(cls_name)-1][2] + 13
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
                g.add_edge(item + 13, items[0])
        for items in self.trees_family_to_species:
            for item in items[1:]:
                g.add_edge(item + 51, items[0] + 13)
        nodes = sorted(g.nodes())
        nodes_idx = dict(zip(nodes, range(len(nodes))))
        g_t = g.reverse()
        am = nx.to_numpy_matrix(g, nodelist=nodes, order=nodes)
        to_eval = [t not in to_skip for t in nodes]
        return g, g_t, np.array(am), to_eval, nodes_idx

class AirDataset(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None, re_level='class', proportion=1.0):
        super(AirDataset, self).__init__()

        self.re_level = re_level
        self.proportion = proportion
        self.trees = [
            [1, 1, 1],
            [2, 2, 1],
            [3, 3, 1],
            [4, 3, 1],
            [5, 3, 1],
            [6, 3, 1],
            [7, 4, 1],
            [8, 4, 1],
            [9, 5, 1],
            [10, 5, 1],
            [11, 5, 1],
            [12, 5, 1],
            [13, 6, 1],
            [14, 7, 2],
            [15, 8, 3],
            [16, 9, 3],
            [17, 10, 7],
            [18, 10, 7],
            [19, 11, 7],
            [20, 12, 4],
            [21, 13, 5],
            [22, 14, 5],
            [23, 15, 5],
            [24, 16, 5],
            [25, 16, 5],
            [26, 16, 5],
            [27, 16, 5],
            [28, 16, 5],
            [29, 16, 5],
            [30, 16, 5],
            [31, 16, 5],
            [32, 17, 5],
            [33, 17, 5],
            [34, 17, 5],
            [35, 17, 5],
            [36, 18, 5],
            [37, 18, 5],
            [38, 19, 5],
            [39, 19, 5],
            [40, 19, 5],
            [41, 20, 5],
            [42, 20, 5],
            [43, 21, 21],
            [44, 22, 14],
            [45, 23, 9],
            [46, 24, 9],
            [47, 25, 9],
            [48, 25, 9],
            [49, 26, 8],
            [50, 27, 8],
            [51, 28, 8],
            [52, 28, 8],
            [53, 29, 12],
            [54, 29, 12],
            [55, 30, 23],
            [56, 31, 14],
            [57, 32, 14],
            [58, 33, 14],
            [59, 34, 23],
            [60, 35, 12],
            [61, 36, 12],
            [62, 37, 12],
            [63, 38, 13],
            [64, 39, 26],
            [65, 40, 15],
            [66, 41, 15],
            [67, 41, 15],
            [68, 41, 15],
            [69, 42, 15],
            [70, 42, 15],
            [71, 43, 15],
            [72, 44, 16],
            [73, 45, 23],
            [74, 46, 22],
            [75, 47, 11],
            [76, 48, 11],
            [77, 49, 18],
            [78, 50, 18],
            [79, 51, 18],
            [80, 52, 6],
            [81, 53, 19],
            [82, 53, 19],
            [83, 54, 7],
            [84, 55, 20],
            [85, 56, 4],
            [86, 57, 21],
            [87, 58, 23],
            [88, 59, 23],
            [89, 59, 23],
            [90, 60, 23],
            [91, 61, 17],
            [92, 62, 25],
            [93, 63, 27],
            [94, 64, 27],
            [95, 65, 28],
            [96, 66, 10],
            [97, 67, 24],
            [98, 68, 29],
            [99, 69, 29],
            [100, 70, 30]
        ]
        self.map = {'A300B4': 1, 'A310': 2, 'A318': 3, 'A319': 4, 'A320': 5, 'A321': 6, 'A330-200': 7, 'A330-300': 8, 'A340-200': 9, 'A340-300': 10, 'A340-500': 11, 'A340-600': 12, 'A380': 13, 'An-12': 14, 'ATR-42': 15, 'ATR-72': 16, 'BAE 146-200': 17, 'BAE 146-300': 18, 'BAE-125': 19, 'Beechcraft 1900': 20, '707-320': 21, 'Boeing 717': 22, '727-200': 23, '737-200': 24, '737-300': 25, '737-400': 26, '737-500': 27, '737-600': 28, '737-700': 29, '737-800': 30, '737-900': 31, '747-100': 32, '747-200': 33, '747-300': 34, '747-400': 35, '757-200': 36, '757-300': 37, '767-200': 38, '767-300': 39, '767-400': 40, '777-200': 41, '777-300': 42, 'C-130': 43, 'C-47': 44, 'Cessna 172': 45, 'Cessna 208': 46, 'Cessna 525': 47, 'Cessna 560': 48, 'Challenger 600': 49, 'CRJ-200': 50, 'CRJ-700': 51, 'CRJ-900': 52, 'DHC-8-100': 53, 'DHC-8-300': 54, 'DC-10': 55, 'DC-3': 56, 'DC-6': 57, 'DC-8': 58, 'DC-9-30': 59, 'DH-82': 60, 'DHC-1': 61, 'DHC-6': 62, 'Dornier 328': 63, 'DR-400': 64, 'EMB-120': 65, 'E-170': 66, 'E-190': 67, 'E-195': 68, 'ERJ 135': 69, 'ERJ 145': 70, 'Embraer Legacy 600': 71, 'Eurofighter Typhoon': 72, 'F/A-18': 73, 'F-16A/B': 74, 'Falcon 2000': 75, 'Falcon 900': 76, 'Fokker 100': 77, 'Fokker 50': 78, 'Fokker 70': 79, 'Global Express': 80, 'Gulfstream IV': 81, 'Gulfstream V': 82, 'Hawk T1': 83, 'Il-76': 84, 'Model B200': 85, 'L-1011': 86, 'MD-11': 87, 'MD-80': 88, 'MD-87': 89, 'MD-90': 90, 'Metroliner': 91, 'PA-28': 92, 'Saab 2000': 93, 'Saab 340': 94, 'Spitfire': 95, 'SR-20': 96, 'Tornado': 97, 'Tu-134': 98, 'Tu-154': 99, 'Yak-42': 100}

        name_list = []
        family_label_list = []
        species_label_list = []

        with open(list_path, 'r') as f:
            for l in f.readlines():
                lists = l.strip().strip('\n').split(' ')
                imagename = lists[0]
                classname = " ".join(i for i in lists[1:])
                name_list.append(imagename + '.jpg')
                class_label = self.map[classname]
                family_label_list.append(self.trees[class_label-1][1] + 30)
                species_label_list.append(class_label + 100)

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
class AirDataset2(data.Dataset):
    def __init__(self, image_dir, list_path, input_transform=None, re_level='class', proportion=1.0):
        super(AirDataset2, self).__init__()

        self.re_level = re_level
        self.proportion = proportion
        self.trees = [
            [1, 1, 1],
            [2, 2, 1],
            [3, 3, 1],
            [4, 3, 1],
            [5, 3, 1],
            [6, 3, 1],
            [7, 4, 1],
            [8, 4, 1],
            [9, 5, 1],
            [10, 5, 1],
            [11, 5, 1],
            [12, 5, 1],
            [13, 6, 1],
            [14, 7, 2],
            [15, 8, 3],
            [16, 9, 3],
            [17, 10, 7],
            [18, 10, 7],
            [19, 11, 7],
            [20, 12, 4],
            [21, 13, 5],
            [22, 14, 5],
            [23, 15, 5],
            [24, 16, 5],
            [25, 16, 5],
            [26, 16, 5],
            [27, 16, 5],
            [28, 16, 5],
            [29, 16, 5],
            [30, 16, 5],
            [31, 16, 5],
            [32, 17, 5],
            [33, 17, 5],
            [34, 17, 5],
            [35, 17, 5],
            [36, 18, 5],
            [37, 18, 5],
            [38, 19, 5],
            [39, 19, 5],
            [40, 19, 5],
            [41, 20, 5],
            [42, 20, 5],
            [43, 21, 21],
            [44, 22, 14],
            [45, 23, 9],
            [46, 24, 9],
            [47, 25, 9],
            [48, 25, 9],
            [49, 26, 8],
            [50, 27, 8],
            [51, 28, 8],
            [52, 28, 8],
            [53, 29, 12],
            [54, 29, 12],
            [55, 30, 23],
            [56, 31, 14],
            [57, 32, 14],
            [58, 33, 14],
            [59, 34, 23],
            [60, 35, 12],
            [61, 36, 12],
            [62, 37, 12],
            [63, 38, 13],
            [64, 39, 26],
            [65, 40, 15],
            [66, 41, 15],
            [67, 41, 15],
            [68, 41, 15],
            [69, 42, 15],
            [70, 42, 15],
            [71, 43, 15],
            [72, 44, 16],
            [73, 45, 23],
            [74, 46, 22],
            [75, 47, 11],
            [76, 48, 11],
            [77, 49, 18],
            [78, 50, 18],
            [79, 51, 18],
            [80, 52, 6],
            [81, 53, 19],
            [82, 53, 19],
            [83, 54, 7],
            [84, 55, 20],
            [85, 56, 4],
            [86, 57, 21],
            [87, 58, 23],
            [88, 59, 23],
            [89, 59, 23],
            [90, 60, 23],
            [91, 61, 17],
            [92, 62, 25],
            [93, 63, 27],
            [94, 64, 27],
            [95, 65, 28],
            [96, 66, 10],
            [97, 67, 24],
            [98, 68, 29],
            [99, 69, 29],
            [100, 70, 30]
        ]
        self.trees_order_to_family = [
            [1, 1, 2, 3, 4, 5, 6], 
            [2, 7], 
            [3, 8, 9], 
            [4, 12, 56], 
            [5, 13, 14, 15, 16, 17, 18, 19, 20], 
            [6, 52], 
            [7, 10, 11, 54], 
            [8, 26, 27, 28], 
            [9, 23, 24, 25], 
            [10, 66], 
            [11, 47, 48], 
            [12, 29, 35, 36, 37], 
            [13, 38], 
            [14, 22, 31, 32, 33], 
            [15, 40, 41, 42, 43], 
            [16, 44], 
            [17, 61], 
            [18, 49, 50, 51], 
            [19, 53], 
            [20, 55], 
            [21, 21, 57], 
            [22, 46], 
            [23, 30, 34, 45, 58, 59, 60], 
            [24, 67], 
            [25, 62], 
            [26, 39], 
            [27, 63, 64], 
            [28, 65], 
            [29, 68, 69], 
            [30, 70]
        ]
        self.trees_family_to_species = [
            [1, 1], 
            [2, 2], 
            [3, 3, 4, 5, 6], 
            [4, 7, 8], 
            [5, 9, 10, 11, 12], 
            [6, 13], 
            [7, 14], 
            [8, 15], 
            [9, 16], 
            [10, 17, 18], 
            [11, 19], 
            [12, 20], 
            [13, 21], 
            [14, 22], 
            [15, 23], 
            [16, 24, 25, 26, 27, 28, 29, 30, 31], 
            [17, 32, 33, 34, 35], 
            [18, 36, 37], 
            [19, 38, 39, 40], 
            [20, 41, 42], 
            [21, 43], 
            [22, 44], 
            [23, 45], 
            [24, 46], 
            [25, 47, 48], 
            [26, 49], 
            [27, 50], 
            [28, 51, 52], 
            [29, 53, 54], 
            [30, 55], 
            [31, 56], 
            [32, 57], 
            [33, 58], 
            [34, 59], 
            [35, 60], 
            [36, 61], 
            [37, 62], 
            [38, 63], 
            [39, 64], 
            [40, 65], 
            [41, 66, 67, 68], 
            [42, 69, 70], 
            [43, 71], 
            [44, 72], 
            [45, 73], 
            [46, 74], 
            [47, 75], 
            [48, 76], 
            [49, 77], 
            [50, 78], 
            [51, 79], 
            [52, 80], 
            [53, 81, 82], 
            [54, 83], 
            [55, 84], 
            [56, 85], 
            [57, 86], 
            [58, 87], 
            [59, 88, 89], 
            [60, 90], 
            [61, 91], 
            [62, 92], 
            [63, 93], 
            [64, 94], 
            [65, 95], 
            [66, 96], 
            [67, 97], 
            [68, 98], 
            [69, 99], 
            [70, 100]
        ]
        self.map = {'A300B4': 1, 'A310': 2, 'A318': 3, 'A319': 4, 'A320': 5, 'A321': 6, 'A330-200': 7, 'A330-300': 8, 'A340-200': 9, 'A340-300': 10, 'A340-500': 11, 'A340-600': 12, 'A380': 13, 'An-12': 14, 'ATR-42': 15, 'ATR-72': 16, 'BAE 146-200': 17, 'BAE 146-300': 18, 'BAE-125': 19, 'Beechcraft 1900': 20, '707-320': 21, 'Boeing 717': 22, '727-200': 23, '737-200': 24, '737-300': 25, '737-400': 26, '737-500': 27, '737-600': 28, '737-700': 29, '737-800': 30, '737-900': 31, '747-100': 32, '747-200': 33, '747-300': 34, '747-400': 35, '757-200': 36, '757-300': 37, '767-200': 38, '767-300': 39, '767-400': 40, '777-200': 41, '777-300': 42, 'C-130': 43, 'C-47': 44, 'Cessna 172': 45, 'Cessna 208': 46, 'Cessna 525': 47, 'Cessna 560': 48, 'Challenger 600': 49, 'CRJ-200': 50, 'CRJ-700': 51, 'CRJ-900': 52, 'DHC-8-100': 53, 'DHC-8-300': 54, 'DC-10': 55, 'DC-3': 56, 'DC-6': 57, 'DC-8': 58, 'DC-9-30': 59, 'DH-82': 60, 'DHC-1': 61, 'DHC-6': 62, 'Dornier 328': 63, 'DR-400': 64, 'EMB-120': 65, 'E-170': 66, 'E-190': 67, 'E-195': 68, 'ERJ 135': 69, 'ERJ 145': 70, 'Embraer Legacy 600': 71, 'Eurofighter Typhoon': 72, 'F/A-18': 73, 'F-16A/B': 74, 'Falcon 2000': 75, 'Falcon 900': 76, 'Fokker 100': 77, 'Fokker 50': 78, 'Fokker 70': 79, 'Global Express': 80, 'Gulfstream IV': 81, 'Gulfstream V': 82, 'Hawk T1': 83, 'Il-76': 84, 'Model B200': 85, 'L-1011': 86, 'MD-11': 87, 'MD-80': 88, 'MD-87': 89, 'MD-90': 90, 'Metroliner': 91, 'PA-28': 92, 'Saab 2000': 93, 'Saab 340': 94, 'Spitfire': 95, 'SR-20': 96, 'Tornado': 97, 'Tu-134': 98, 'Tu-154': 99, 'Yak-42': 100}

        self.g, self.g_t, self.adj_matrix, self.to_eval, self.nodes_idx = self.compute_adj_matrix()

        class_imgs = {}
        with open(list_path, 'r') as f:
            for l in f.readlines():
                lists = l.strip().strip('\n').split(' ')
                imagename = lists[0]
                classname = " ".join(i for i in lists[1:])
                class_label = self.map[classname]
                y_ = np.zeros(201)
                y_[[self.nodes_idx.get(a) for a in nx.ancestors(self.g_t, class_label + 100)]] = 1
                y_[self.nodes_idx[class_label + 100]] = 1
                if class_label not in class_imgs.keys():
                    class_imgs[class_label] = {'imgs': [], 'labels': []}
                class_imgs[class_label]['imgs'].append(join(image_dir, imagename + '.jpg'))
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
                y_ = np.zeros(201)
                family = self.trees[key-1][1] + 30
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
                g.add_edge(item + 30, items[0])
        for items in self.trees_family_to_species:
            for item in items[1:]:
                g.add_edge(item + 100, items[0] + 30)
        nodes = sorted(g.nodes())
        nodes_idx = dict(zip(nodes, range(len(nodes))))
        g_t = g.reverse()
        am = nx.to_numpy_matrix(g, nodelist=nodes, order=nodes)
        to_eval = [t not in to_skip for t in nodes]
        return g, g_t, np.array(am), to_eval, nodes_idx
