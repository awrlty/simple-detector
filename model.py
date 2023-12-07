import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary

import config


class CornerDetectionNet(nn.Module):
    def __init__(self):
        super(CornerDetectionNet, self).__init__()

        self.feature_size = config.S
        self.num_bboxes = config.B

        self.backbone = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.backbone.requires_grad_(False)

        self.detector = self._make_detector()

    def _make_detector(self):
        net = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False),
            nn.Conv2d(512, 256, kernel_size=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 10, kernel_size=1),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        return net

    def forward(self, x):  # input shape: 3, 1024, 1024
        x = self.backbone(x)  # output shape: 6, 34, 34
        x = self.detector(x)  # output shape: 6, 32, 32

        x = x.view(-1, self.feature_size, self.feature_size, 5 * self.num_bboxes)
        x = nn.Sigmoid()(x)
        return x


def test():
    # import cv2
    # import numpy as np
    # import torchvision.transforms as transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CornerDetectionNet().to(device)
    # summary(model, input_size=(3, config.IMAGE_SIZE, config.IMAGE_SIZE))

    image = torch.rand(config.BATCH_SIZE, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)
    # image = r"D:\Projects\2023_daqs_exterior_wall_quality_inspector\dataset\class_labeling\opening_corner\images\DJI_20210928091447_0018_SUPR_X3238Y1633.jpg"
    # image = cv2.imdecode(np.fromfile(image, np.uint8), cv2.IMREAD_UNCHANGED)
    # orig_image = image.copy()

    # image = transforms.ToTensor()(image).unsqueeze(0).to(device)
    output = model(image)

    # Check ouput tensor size, which should be [B, S, S, depth]
    print(output.shape)
    # print(torch.nonzero(output.view(-1)).size(0))


def test_nan():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CornerDetectionNet().to(device)

    for param in model.parameters():
        if torch.isnan(param).any():
            print("Nan detected: ", param)


if __name__ == '__main__':
    test()
    # test_nan()
