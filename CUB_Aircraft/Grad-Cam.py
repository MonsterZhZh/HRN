# coding: utf-8
import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from dataset import CubDataset
from tree_loss import TreeLoss
from utils import get_order_family_target


def tensor_to_cv2(tensor):
    tensor = tensor / 2 + 0.5
    array = tensor.cpu().squeeze().numpy()
    maxValue = array.max()
    array = array * 255 / maxValue
    mat = np.uint8(array)
    mat = mat.transpose(1, 2, 0)
    img = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    return img


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir, idx, cls_idx):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, "cam_"+"_"+str(cls_idx)+"_"+str(idx)+".jpg")
    path_raw_img = os.path.join(out_dir, "raw_"+"_"+str(cls_idx)+"_"+str(idx)+".jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))


def gen_cam(feature_map, grads):
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (448, 448))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':
    batch_size = 1

    transform_test = transforms.Compose([
            transforms.Resize((550, 550)),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    data_dir = '/home/datasets/HI_Datasets/CUB2011/CUB_200_2011/images'
    test_list = '/home/datasets/HI_Datasets/CUB2011/CUB_200_2011/hierarchy/test_images_4_level_V1.txt'
    test_dir = '/home/datasets/HI_Datasets/CUB2011/CUB_200_2011/test'
    trees = [
            [51, 11, 47],
            [52, 11, 47],
            [53, 11, 47],
            [54, 5, 21],
            [55, 3, 16],
            [56, 3, 16],
            [57, 3, 16],
            [58, 3, 16],
            [59, 7, 30],
            [60, 7, 30],
            [61, 7, 30],
            [62, 7, 30],
            [63, 7, 30],
            [64, 7, 25],
            [65, 7, 25],
            [66, 7, 25],
            [67, 7, 25],
            [68, 7, 38],
            [69, 7, 33],
            [70, 7, 31],
            [71, 7, 36],
            [72, 2, 15],
            [73, 12, 49],
            [74, 12, 49],
            [75, 12, 49],
            [76, 7, 30],
            [77, 7, 30],
            [78, 7, 26],
            [79, 7, 27],
            [80, 7, 27],
            [81, 5, 21],
            [82, 5, 21],
            [83, 5, 21],
            [84, 7, 28],
            [85, 7, 28],
            [86, 9, 45],
            [87, 7, 42],
            [88, 7, 42],
            [89, 7, 42],
            [90, 7, 42],
            [91, 7, 42],
            [92, 7, 42],
            [93, 7, 42],
            [94, 12, 50],
            [95, 11, 48],
            [96, 0, 13],
            [97, 7, 28],
            [98, 7, 28],
            [99, 7, 30],
            [100, 10, 46],
            [101, 10, 46],
            [102, 10, 46],
            [103, 10, 46],
            [104, 7, 25],
            [105, 7, 28],
            [106, 7, 28],
            [107, 7, 25],
            [108, 3, 16],
            [109, 3, 17],
            [110, 3, 17],
            [111, 3, 17],
            [112, 3, 17],
            [113, 3, 17],
            [114, 3, 17],
            [115, 3, 17],
            [116, 3, 17],
            [117, 1, 14],
            [118, 1, 14],
            [119, 1, 14],
            [120, 1, 14],
            [121, 3, 18],
            [122, 3, 18],
            [123, 7, 27],
            [124, 7, 27],
            [125, 7, 27],
            [126, 7, 36],
            [127, 7, 42],
            [128, 7, 42],
            [129, 4, 19],
            [130, 4, 19],
            [131, 4, 19],
            [132, 4, 19],
            [133, 4, 19],
            [134, 4, 20],
            [135, 7, 23],
            [136, 6, 22],
            [137, 0, 13],
            [138, 7, 30],
            [139, 0, 13],
            [140, 0, 13],
            [141, 7, 33],
            [142, 2, 15],
            [143, 7, 27],
            [144, 7, 39],
            [145, 7, 30],
            [146, 7, 30],
            [147, 7, 30],
            [148, 7, 30],
            [149, 7, 35],
            [150, 8, 44],
            [151, 8, 44],
            [152, 7, 42],
            [153, 7, 42],
            [154, 7, 34],
            [155, 2, 15],
            [156, 3, 16],
            [157, 7, 27],
            [158, 7, 27],
            [159, 7, 35],
            [160, 5, 21],
            [161, 7, 32],
            [162, 7, 32],
            [163, 7, 36],
            [164, 7, 36],
            [165, 7, 36],
            [166, 7, 36],
            [167, 7, 36],
            [168, 7, 37],
            [169, 7, 36],
            [170, 7, 36],
            [171, 7, 36],
            [172, 7, 36],
            [173, 7, 36],
            [174, 7, 36],
            [175, 7, 36],
            [176, 7, 36],
            [177, 7, 36],
            [178, 7, 36],
            [179, 7, 36],
            [180, 7, 36],
            [181, 7, 36],
            [182, 7, 36],
            [183, 7, 36],
            [184, 7, 40],
            [185, 7, 29],
            [186, 7, 29],
            [187, 7, 29],
            [188, 7, 29],
            [189, 7, 25],
            [190, 7, 25],
            [191, 3, 17],
            [192, 3, 17],
            [193, 3, 17],
            [194, 3, 17],
            [195, 3, 17],
            [196, 3, 17],
            [197, 3, 17],
            [198, 7, 36],
            [199, 7, 33],
            [200, 7, 33],
            [201, 7, 43],
            [202, 7, 43],
            [203, 7, 43],
            [204, 7, 43],
            [205, 7, 43],
            [206, 7, 43],
            [207, 7, 43],
            [208, 7, 35],
            [209, 7, 35],
            [210, 7, 35],
            [211, 7, 35],
            [212, 7, 35],
            [213, 7, 35],
            [214, 7, 35],
            [215, 7, 35],
            [216, 7, 35],
            [217, 7, 35],
            [218, 7, 35],
            [219, 7, 35],
            [220, 7, 35],
            [221, 7, 35],
            [222, 7, 35],
            [223, 7, 35],
            [224, 7, 35],
            [225, 7, 35],
            [226, 7, 35],
            [227, 7, 35],
            [228, 7, 35],
            [229, 7, 35],
            [230, 7, 35],
            [231, 7, 35],
            [232, 7, 35],
            [233, 7, 35],
            [234, 7, 35],
            [235, 7, 24],
            [236, 7, 24],
            [237, 9, 45],
            [238, 9, 45],
            [239, 9, 45],
            [240, 9, 45],
            [241, 9, 45],
            [242, 9, 45],
            [243, 7, 41],
            [244, 7, 41],
            [245, 7, 41],
            [246, 7, 41],
            [247, 7, 41],
            [248, 7, 41],
            [249, 7, 41],
            [250, 7, 35]
            ]
    levels = 3
    total_nodes = 251

    testset = CubDataset(data_dir, test_list, transform_test, re_level='class', proportion=1.0)
    # testset = CubDataset2(test_dir, transform_test, 'class', 1.0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

    device = torch.device("cuda:5")
    net = torch.load('./models_CUB/model_CUB_200_448_p1.0_bz8_Tree_CE_Cos.pth')
    net.to(device)
    net.eval()

    fmap_block = list()
    grad_block = list()

    # register hook
    net.conv_block3.register_forward_hook(farward_hook)
    net.conv_block3.register_backward_hook(backward_hook)
        
    CELoss = nn.CrossEntropyLoss()
    tree_loss = TreeLoss(trees, total_nodes, levels, device)

    idx  = 60
        
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx >= idx:
            break

        inputs, targets = inputs.to(device), targets.to(device)
        order_targets, family_targets, target_list_sig = get_order_family_target(targets, device, 'CUB')
        leaf_labels = torch.nonzero(targets > 50, as_tuple=False)
        select_leaf_labels = torch.index_select(targets, 0, leaf_labels.squeeze()) - 51
        
        # forward
        xc1_sig, xc2_sig, xc3, xc3_sig = net(inputs)

        # backward
        net.zero_grad()
        # xc1_sig[0, order_targets][0].backward()
        # xc2_sig[0, family_targets][0].backward()
        # xc3_sig[0, select_leaf_labels][0].backward()
        xc3[0, select_leaf_labels][0].backward()

        # generate cam
        grads_val = grad_block[batch_idx].cpu().data.numpy().squeeze()
        fmap = fmap_block[batch_idx].cpu().data.numpy().squeeze()
        cam = gen_cam(fmap, grads_val)

        # save cam picture
        img = tensor_to_cv2(inputs)
        img_show = np.float32(cv2.resize(img, (448, 448))) / 255
        cls_idx = int(select_leaf_labels[0]) + 1
        show_cam_on_image(img_show, cam, './vis_specie_ce/', batch_idx, cls_idx)














