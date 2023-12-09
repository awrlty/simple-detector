import math
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config


class PointDataset(Dataset):
    def __init__(self, set_type, augment=False):
        assert set_type in {'train', 'valid'}
        with open(os.path.join(config.DATA_PATH, f"{set_type}.txt"), "r") as f:
            self.filenames = f.read().strip().split("\n")

        self.image_dir = os.path.join(config.DATA_PATH, "images")
        self.augment = augment
        self.transform = transforms.ColorJitter(brightness=0.3, contrast=0.3)

        self.boxes = []
        # label_path = os.path.join(config.DATA_PATH, "labels_voc_format.txt")
        label_path = os.path.join(config.DATA_PATH, "labels_integrated.txt")  # yolo format
        with open(label_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            splitted = line.strip().split()

            if len(splitted) == 1:  # line with no box information
                self.boxes.append(torch.zeros((1, 4)))
                continue

            num_boxes = (len(splitted) - 1) // 5  # 맨 앞 파일명 제외하고 box(x, y, w, h, conf) 개수
            box, label = [], []
            for i in range(num_boxes):
                x = float(splitted[5 * i + 1])
                y = float(splitted[5 * i + 2])
                w = float(splitted[5 * i + 3])
                h = float(splitted[5 * i + 4])
                box.append([x, y, w, h])
            self.boxes.append(torch.Tensor(box))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Resize and Augment Image
        image_path = os.path.join(self.image_dir, f"{self.filenames[idx]}.jpg")
        image = Image.open(image_path).convert('RGB')
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        if self.augment:
            image = self.transform(image)
        image = transforms.ToTensor()(image)

        boxes = self.boxes[idx].clone()  # [N, 4]; N is the number of labeled object (x, y, w, h)
        target = self.get_label_matrix(boxes)
        return image, target

    def get_label_matrix(self, boxes):  # [N, 4]
        """ Encode box coordinates and class labels as one target tensor.
        Args:
            boxes: (tensor) [[x, y, w, h]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image width/height. == yolo data format
        Returns:
            An encoded tensor sized [S, S, 5 * B], 5=(x, y, w, h, conf)
        """
        depth = 5 * config.B
        target = torch.zeros(config.S, config.S, depth)
        if boxes.sum().item() == 0.0:  # label with no objects
            return target

        for box in boxes:
            x, y, w, h = box.tolist()
            i, j = math.ceil(y * config.IMAGE_SIZE / config.S) - 1, math.ceil(x * config.IMAGE_SIZE / config.S) - 1
            x0, y0 = (j * config.S), (i * config.S)
            x_cell, y_cell = (x * config.IMAGE_SIZE - x0) / config.S, (y * config.IMAGE_SIZE - y0) / config.S

            target[j, i, :4] = torch.tensor([x_cell, y_cell, w, h])
            target[j, i, 4] = 1.0
        return target


class PointDetectDatset(Dataset):
    def __init__(self):
        self.images = [file for file in os.listdir(config.TEST_PATH) if file.lower().endswith(".jpg")]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(config.TEST_PATH, self.images[idx])
        image = Image.open(image_path).convert('RGB')
        image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE))
        image = transforms.ToTensor()(image)

        filename = self.images[idx]
        return image, filename


def check_nan(idx, images, labels):
    nan_img = torch.isnan(images).int().sum().item()
    nan_label = torch.isnan(labels).int().sum().item()

    print(f"{idx + 1}:")
    print(f"Nan of images: {nan_img}")
    print(f"Nan of labels: {nan_label}\n")


def decode_labels(labels):
    decoded_labels = []
    for label in labels:
        for i in range(config.S):
            for j in range(config.S):
                x_cell, y_cell, w, h, conf = label[j, i, :].tolist()[:5]
                if conf == 1.0:
                    decoded_labels.append([(j * config.S) + x_cell * config.S, (i * config.S) + (y_cell * config.S), w, h])
                    print([((j * config.S) + x_cell * config.S)/config.IMAGE_SIZE, ((i * config.S) + (y_cell * config.S))/config.IMAGE_SIZE, w, h])


if __name__ == "__main__":
    train_set = PointDataset('train', augment=False)
    train_loader = DataLoader(train_set,
                              batch_size=config.BATCH_SIZE,
                              # batch_size=1,
                              num_workers=config.NUM_WORKERS,
                              persistent_workers=False,
                              drop_last=True,
                              shuffle=False)

    for idx, (images, labels) in enumerate(train_loader):
        decode_labels(labels)

        # if torch.sum(labels.squeeze(0)) == 0.0:
        #     print(idx)

        # print(torch.nonzero(labels.view(-1)).size(0))
        # print("end")
        # break
        # check_nan(idx, images, labels)
