from PIL import Image
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import transforms, models
import torch.hub
import argparse
from torch.optim import lr_scheduler

from RFM_car import HIFD2

from tree_loss import TreeLoss
from dataset import CarDataset, CarDataset2
from train_test_car import test_AP, train, test


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch Deployment')
    parser.add_argument('--worker', default=4, type=int, help='number of workers (default: 4)')
    parser.add_argument('--model', type=str, default='./pre-trained/resnet50-19c8e357.pth', help='Path of pre-trained model')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
    parser.add_argument('--proportion', type=float, default=1.0, help='Proportion of species label')  
    parser.add_argument('--epoch', default=200, type=int, help='Epochs')
    parser.add_argument('--batch', type=int, help='batch size')      
    parser.add_argument('--dataset', type=str, default='Car', help='dataset name')
    parser.add_argument('--img_size', type=str, default='448', help='image size')
    parser.add_argument('--lr_adjt', type=str, default='Cos', help='Learning rate schedual', choices=['Cos', 'Step'])
    parser.add_argument('--device', nargs='+', default='2', help='GPU IDs for DP training')
    args = parser.parse_args()
    
    if args.proportion == 0.1: 
        args.epoch = 100
        args.batch = 32
        args.lr_adjt = 'Step'
    
    return args


if __name__ == '__main__':
    args = arg_parse()
    print('==> proportion: ', args.proportion)
    print('==> epoch: ', args.epoch)
    print('==> batch: ', args.batch)
    print('==> dataset: ', args.dataset)
    print('==> img_size: ', args.img_size)
    print('==> device: ', args.device)
    print('==> Schedual: ', args.lr_adjt)

    nb_epoch = args.epoch
    batch_size = args.batch
    num_workers = args.worker

    # Preprocess
    transform_train = transforms.Compose([
    transforms.Resize((550, 550)),
    transforms.RandomCrop(448, padding=8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Data
    if args.dataset == 'Car':
        # StandCars
        data_dir = '/home/datasets/HI_Datasets/StandCars/cars196'
        train_list = '/home/datasets/HI_Datasets/StandCars/car_train.txt'
        test_list = '/home/datasets/HI_Datasets/StandCars/car_test.txt'
        trees = [
            [9, 6],
            [10, 5],
            [11, 5],
            [12, 5],
            [13, 5],
            [14, 2],
            [15, 3],
            [16, 1],
            [17, 2],
            [18, 1],
            [19, 2],
            [20, 1],
            [21, 2],
            [22, 2],
            [23, 2],
            [24, 5],
            [25, 5],
            [26, 8],
            [27, 3],
            [28, 5],
            [29, 1],
            [30, 2],
            [31, 5],
            [32, 5],
            [33, 2],
            [34, 5],
            [35, 1],
            [36, 2],
            [37, 5],
            [38, 8],
            [39, 1],
            [40, 6],
            [41, 6],
            [42, 2],
            [43, 5],
            [44, 1],
            [45, 6],
            [46, 1],
            [47, 1],
            [48, 5],
            [49, 5],
            [50, 2],
            [51, 2],
            [52, 5],
            [53, 1],
            [54, 2],
            [55, 5],
            [56, 6],
            [57, 5],
            [58, 6],
            [59, 5],
            [60, 6],
            [61, 0],
            [62, 0],
            [63, 1],
            [64, 2],
            [65, 2],
            [66, 6],
            [67, 1],
            [68, 4],
            [69, 5],
            [70, 6],
            [71, 5],
            [72, 7],
            [73, 0],
            [74, 2],
            [75, 5],
            [76, 6],
            [77, 0],
            [78, 0],
            [79, 7],
            [80, 2],
            [81, 5],
            [82, 0],
            [83, 0],
            [84, 6],
            [85, 1],
            [86, 4],
            [87, 5],
            [88, 1],
            [89, 1],
            [90, 8],
            [91, 8],
            [92, 8],
            [93, 4],
            [94, 0],
            [95, 0],
            [96, 7],
            [97, 6],
            [98, 0],
            [99, 0],
            [100, 8],
            [101, 2],
            [102, 6],
            [103, 6],
            [104, 5],
            [105, 5],
            [106, 3],
            [107, 2],
            [108, 1],
            [109, 2],
            [110, 1],
            [111, 1],
            [112, 2],
            [113, 5],
            [114, 0],
            [115, 1],
            [116, 4],
            [117, 6],
            [118, 6],
            [119, 0],
            [120, 2],
            [121, 0],
            [122, 0],
            [123, 5],
            [124, 8],
            [125, 5],
            [126, 6],
            [127, 7],
            [128, 6],
            [129, 6],
            [130, 0],
            [131, 1],
            [132, 0],
            [133, 0],
            [134, 4],
            [135, 4],
            [136, 2],
            [137, 5],
            [138, 3],
            [139, 6],
            [140, 6],
            [141, 6],
            [142, 5],
            [143, 5],
            [144, 5],
            [145, 5],
            [146, 5],
            [147, 3],
            [148, 5],
            [149, 2],
            [150, 6],
            [151, 6],
            [152, 2],
            [153, 6],
            [154, 6],
            [155, 6],
            [156, 6],
            [157, 6],
            [158, 2],
            [159, 2],
            [160, 2],
            [161, 2],
            [162, 6],
            [163, 6],
            [164, 5],
            [165, 1],
            [166, 1],
            [167, 6],
            [168, 2],
            [169, 1],
            [170, 5],
            [171, 2],
            [172, 5],
            [173, 5],
            [174, 7],
            [175, 5],
            [176, 3],
            [177, 7],
            [178, 3],
            [179, 2],
            [180, 2],
            [181, 5],
            [182, 4],
            [183, 1],
            [184, 5],
            [185, 5],
            [186, 3],
            [187, 1],
            [188, 2],
            [189, 5],
            [190, 5],
            [191, 3],
            [192, 5],
            [193, 5],
            [194, 6],
            [195, 5],
            [196, 5],
            [197, 6],
            [198, 3],
            [199, 3],
            [200, 3],
            [201, 3],
            [202, 5],
            [203, 6],
            [204, 1]
        ]
        levels = 2
        total_nodes = 205
        trainset = CarDataset(data_dir, train_list, transform_train, re_level='family', proportion=args.proportion)
        # Uncomment this line for testing OA results
        testset = CarDataset(data_dir, test_list, transform_test, re_level='class', proportion=1.0)
        # Uncomment this line for testing Average PRC results
        # testset = CarDataset2(data_dir, test_list, transform_test, re_level='class', proportion=1.0)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    # GPU
    device = torch.device("cuda:" + args.device[0])
    
    # RFM
    backbone = models.resnet50(pretrained=False)
    backbone.load_state_dict(torch.load('../pre-trained/resnet50-19c8e357.pth'))
    # backbone = models.resnext101_32x8d(pretrained=False)
    # backbone.load_state_dict(torch.load('../pre-trained/resnext101_32x8d-8ba56ff5.pth'))
    net = HIFD2(args.dataset, backbone, 1024)

    # RFM from trained model
    # net = torch.load(args.model)

    net.to(device)

    # Loss functions
    CELoss = nn.CrossEntropyLoss()
    tree = TreeLoss(trees, total_nodes, levels, device)

    # Layers for HIFD
    if args.proportion > 0.1:       # for p > 0.1
        optimizer = optim.SGD([
            {'params': net.classifier_1.parameters(), 'lr': 0.002},
            {'params': net.classifier_2.parameters(), 'lr': 0.002},
            {'params': net.classifier_2_1.parameters(), 'lr': 0.002},
            {'params': net.fc1.parameters(), 'lr': 0.002},
            {'params': net.fc2.parameters(), 'lr': 0.002},
            {'params': net.conv_block1.parameters(), 'lr': 0.002},
            {'params': net.conv_block2.parameters(), 'lr': 0.002},
            {'params': net.features.parameters(), 'lr': 0.0002}
        ],
        momentum=0.9, weight_decay=5e-4)
    
    else:     # for p = 0.1
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)

    save_name = args.dataset+'_'+str(args.epoch)+'_'+str(args.batch)+'_'+str(args.img_size)+'_'+str(args.proportion)+'_ResNet-50_'+'_'+args.lr_adjt
    train(nb_epoch, net, trainloader, testloader, optimizer, scheduler, args.lr_adjt, CELoss, tree, device, args.device, save_name, args.dataset)
    
    # Evaluate OA
    # test(net, testloader, CELoss, tree, device, args.dataset)

    # Evaluate Average PRC
    # test_AP(net, args.dataset, testset, testloader, device)
