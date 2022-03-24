import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # Post-activation
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None
        # Pre-activation
        # self.bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        # self.relu = nn.ReLU() if relu else None
        # self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
        #                       stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        # x = self.conv(x)
        return x


class HIFD2(nn.Module):
    def __init__(self, dataset, model, feature_size):
        super(HIFD2, self).__init__()

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.num_ftrs = 2048 * 1 * 1

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        if dataset == 'Car':
            self.classifier_1 = nn.Sequential(
                    nn.Linear(512, 9),
                    nn.Sigmoid()
                )
            self.classifier_2 = nn.Sequential(
                    nn.Linear(512, 196),
                    nn.Sigmoid()
                )
            self.classifier_2_1 = nn.Sequential(
                    nn.Linear(512, 196),
                )


    def forward(self, x):
        x = self.features(x)
        x_order = self.conv_block1(x)
        x_species = self.conv_block2(x)

        x_order_fc = self.pooling(x_order)
        x_order_fc = x_order_fc.view(x_order_fc.size(0), -1)
        x_order_fc = self.fc1(x_order_fc)

        x_species_fc = self.pooling(x_species)
        x_species_fc = x_species_fc.view(x_species_fc.size(0), -1)
        x_species_fc = self.fc2(x_species_fc)

        y_order_sig = self.classifier_1(self.relu(x_order_fc))
        y_species_sig = self.classifier_2(self.relu(x_species_fc + x_order_fc))
        y_species_sof = self.classifier_2_1(self.relu(x_species_fc + x_order_fc))

        return y_order_sig, y_species_sof, y_species_sig
    