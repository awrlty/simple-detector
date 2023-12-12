import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import PointDetectDatset
from model import CornerDetectionNet


def visualize_boxes(image_bgr, boxes, class_names, probs, name_bgr_dict=None, line_thickness=2):
    image_boxes = image_bgr.copy()
    for box, class_name, prob in zip(boxes, class_names, probs):
        # Draw box on the image.
        left_top, right_bottom = box
        left, top = int(left_top[0]), int(left_top[1])
        right, bottom = int(right_bottom[0]), int(right_bottom[1])
        bgr = name_bgr_dict[class_name]
        cv2.rectangle(image_boxes, (left, top), (right, bottom), bgr, thickness=line_thickness)

        # Draw text on the image.
        text = '%s %.2f' % (class_name, prob)
        size, baseline = cv2.getTextSize(text,  cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
        text_w, text_h = size

        x, y = left, top
        x1y1 = (x, y)
        x2y2 = (x + text_w + line_thickness, y + text_h + line_thickness + baseline)
        cv2.rectangle(image_boxes, x1y1, x2y2, bgr, -1)
        cv2.putText(image_boxes, text, (x + line_thickness, y + 2*baseline + line_thickness),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255), thickness=1, lineType=8)

    return image_boxes


def decode_preds(pred_tensor, conf_thresh=0.3):
    pred_tensor = pred_tensor.cpu().squeeze(0)
    boxes, confidences = [], []

    cell_size = 1.0 / config.S

    for i in range(config.S):  # for x-dimension
        for j in range(config.S):  # for y-dimension
            for b in range(config.B):
                conf = pred_tensor[j, i, 5 * b + 4]
                print(conf)
                if float(conf) < conf_thresh:
                    continue

                # Compute box corner (x1, y1, x2, y2) from tensor
                box = pred_tensor[j, i, 5 * b: 5 * b + 4]
                x0y0_normalized = torch.FloatTensor([i, j]) * cell_size
                xy_normalized = box[:2] * cell_size + x0y0_normalized
                wh_normalized = box[2:]

                box_xyxy = torch.FloatTensor(4)  # [4,]
                box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized  # left-top corner (x1, y1)
                box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized  # right-bottom corner (x2, y2)

                boxes.append(box_xyxy)
                confidences.append(conf)

    if len(boxes):
        boxes = torch.stack(boxes, 0)  # [n_boxes, 4]
        confidences = torch.stack(confidences, 0)
    else:
        boxes = torch.FloatTensor(0, 4)
        confidences = torch.FloatTensor(0)

    return boxes, confidences





# def post_process_pred(pred_tensor, conf_thresh=0.3, nms_thresh=0.35):


class CornerDetector:
    def __init__(self, conf_thresh=0.3, prob_thresh=0.1, nms_thresh=0.5):
        self.net = CornerDetectionNet().to(device).eval()

        self.S = self.net.feature_size
        self.B = self.net.num_bboxes
        self.C = self.net.num_classes

        self.conf_thresh = conf_thresh
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

    def detect(self, image_bgr, image_size=1024):
        """ Detect objects from given image.
        Args:
            image_bgr: (numpy array) input image in BGR ids_sorted, sized [h, w, 3].
            image_size: (int) image width and height to which input image is resized.
        Returns:
            boxes_detected: (list of tuple) box corner list like [((x1, y1), (x2, y2))_obj1, ...]. Re-scaled for original input image size.
            class_names_detected: (list of str) list of class name for each detected boxe.
            probs_detected: (list of float) list of probability(=confidence x class_score) for each detected box.
        """

        with torch.no_grad():
            pred_tensor = self.net(image_bgr)
        pred_tensor = pred_tensor.cpu().data
        pred_tensor = pred_tensor.squeeze(0)  # squeeze batch dimension.

        # Get detected boxes_detected, labels, confidences, class-scores.
        boxes_normalized_all, class_labels_all, confidences_all, class_scores_all = self.decode(pred_tensor)
        if boxes_normalized_all.size(0) == 0:
            return [], [], []  # if no box found, return empty lists.

        # Apply non maximum supression for boxes of each class.
        boxes_normalized, class_labels, probs = [], [], []

        for class_label in range(len(self.class_name_list)):
            mask = (class_labels_all == class_label)
            if torch.sum(mask) == 0:
                continue  # if no box found, skip that class.

            boxes_normalized_masked = boxes_normalized_all[mask]
            class_labels_maked = class_labels_all[mask]
            confidences_masked = confidences_all[mask]
            class_scores_masked = class_scores_all[mask]

            ids = self.nms(boxes_normalized_masked, confidences_masked)

            boxes_normalized.append(boxes_normalized_masked[ids])
            class_labels.append(class_labels_maked[ids])
            probs.append(confidences_masked[ids] * class_scores_masked[ids])

        boxes_normalized = torch.cat(boxes_normalized, 0)
        class_labels = torch.cat(class_labels, 0)
        probs = torch.cat(probs, 0)

        # Postprocess for box, labels, probs.
        boxes_detected, class_names_detected, probs_detected = [], [], []
        for b in range(boxes_normalized.size(0)):
            box_normalized = boxes_normalized[b]
            class_label = class_labels[b]
            prob = probs[b]

            x1, x2 = 20/1024 * box_normalized[0], 20/1024 * box_normalized[2]  # unnormalize x with image width.
            y1, y2 = 20/1024 * box_normalized[1], 20/1024 * box_normalized[3]  # unnormalize y with image height.
            boxes_detected.append(((x1, y1), (x2, y2)))

            class_label = int(class_label)  # convert from LongTensor to int.
            class_name = self.class_name_list[class_label]
            class_names_detected.append(class_name)

            prob = float(prob)  # convert from Tensor to float.
            probs_detected.append(prob)

        return boxes_detected, class_names_detected, probs_detected

    def decode(self, pred_tensor):
        """ Decode tensor into box coordinates, class labels, and probs_detected.
        Args:
            pred_tensor: (tensor) tensor to decode sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        Returns:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...]. Normalized from 0.0 to 1.0 w.r.t. image width/height, sized [n_boxes, 4].
            labels: (tensor) class labels for each detected boxe, sized [n_boxes,].
            confidences: (tensor) objectness confidences for each detected box, sized [n_boxes,].
            class_scores: (tensor) scores for most likely class for each detected box, sized [n_boxes,].
        """
        S, B, C = self.S, self.B, self.C
        boxes, labels, confidences, class_scores = [], [], [], []

        cell_size = 1.0 / float(S)

        conf = pred_tensor[:, :, 4].unsqueeze(2)  # [S, S, 1]
        for b in range(1, B):
            conf = torch.cat((conf, pred_tensor[:, :, 5*b + 4].unsqueeze(2)), 2)
        conf_mask = conf > self.conf_thresh  # [S, S, B]

        # TBM, further optimization may be possible by replacing the following for-loops with tensor operations.
        for i in range(S):  # for x-dimension.
            for j in range(S):  # for y-dimension.
                class_score, class_label = torch.max(pred_tensor[j, i, 5*B:], 0)

                for b in range(B):
                    conf = pred_tensor[j, i, 5*b + 4]
                    prob = conf * class_score
                    if float(prob) < self.prob_thresh:
                        continue

                    # Compute box corner (x1, y1, x2, y2) from tensor.
                    box = pred_tensor[j, i, 5*b : 5*b + 4]
                    x0y0_normalized = torch.FloatTensor([i, j]) * cell_size  # cell left-top corner. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    xy_normalized = box[:2] * cell_size + x0y0_normalized   # box center. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    wh_normalized = box[2:]  # Box width and height. Normalized from 0.0 to 1.0 w.r.t. image width/height.
                    box_xyxy = torch.FloatTensor(4)  # [4,]
                    box_xyxy[:2] = xy_normalized - 0.5 * wh_normalized  # left-top corner (x1, y1).
                    box_xyxy[2:] = xy_normalized + 0.5 * wh_normalized  # right-bottom corner (x2, y2).

                    # Append result to the lists.
                    boxes.append(box_xyxy)
                    labels.append(class_label)
                    confidences.append(conf)
                    class_scores.append(class_score)

        if len(boxes) > 0:
            boxes = torch.stack(boxes, 0)  # [n_boxes, 4]
            labels = torch.stack(labels, 0)             # [n_boxes, ]
            confidences = torch.stack(confidences, 0)   # [n_boxes, ]
            class_scores = torch.stack(class_scores, 0)  # [n_boxes, ]
        else:
            # If no box found, return empty tensors.
            boxes = torch.FloatTensor(0, 4)
            labels = torch.LongTensor(0)
            confidences = torch.FloatTensor(0)
            class_scores = torch.FloatTensor(0)

        return boxes, labels, confidences, class_scores

    def nms(self, boxes, scores):
        """ Apply non maximum supression.
        Args:
        Returns:
        """
        threshold = self.nms_thresh

        x1 = boxes[:, 0]  # [n,]
        y1 = boxes[:, 1]  # [n,]
        x2 = boxes[:, 2]  # [n,]
        y2 = boxes[:, 3]  # [n,]
        areas = (x2 - x1) * (y2 - y1)  # [n,]

        _, ids_sorted = scores.sort(0, descending=True)  # [n,]
        ids = []
        while ids_sorted.numel() > 0:
            # Assume `ids_sorted` size is [m,] in the beginning of this iter.

            i = ids_sorted.item() if (ids_sorted.numel() == 1) else ids_sorted[0]
            ids.append(i)

            if ids_sorted.numel() == 1:
                break # If only one box is left (i.e., no box to supress), break.

            inter_x1 = x1[ids_sorted[1:]].clamp(min=x1[i])  # [m-1, ]
            inter_y1 = y1[ids_sorted[1:]].clamp(min=y1[i])  # [m-1, ]
            inter_x2 = x2[ids_sorted[1:]].clamp(max=x2[i])  # [m-1, ]
            inter_y2 = y2[ids_sorted[1:]].clamp(max=y2[i])  # [m-1, ]
            inter_w = (inter_x2 - inter_x1).clamp(min=0)  # [m-1, ]
            inter_h = (inter_y2 - inter_y1).clamp(min=0)  # [m-1, ]

            inters = inter_w * inter_h  # intersections b/w/ box `i` and other boxes, sized [m-1, ].
            unions = areas[i] + areas[ids_sorted[1:]] - inters  # unions b/w/ box `i` and other boxes, sized [m-1, ].
            ious = inters / unions  # [m-1, ]

            # Remove boxes whose IoU is higher than the threshold.
            ids_keep = (ious <= threshold).nonzero().squeeze()  # [m-1, ]. Because `nonzero()` adds extra dimension, squeeze it.
            if ids_keep.numel() == 0:
                break  # If no box left, break.
            ids_sorted = ids_sorted[ids_keep+1]  # `+1` is needed because `ids_sorted[0] = i`.

        return torch.LongTensor(ids)


if __name__ == '__main__':
    testset = PointDetectDatset()
    test_loader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=8)  # batched inference는 추후 구현
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r"D:\Projects\2023_daqs_exterior_wall_quality_inspector\corner-detector\results\corner_detector\Dec12_14-33-17\model_best.pth"
    model = CornerDetectionNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    with torch.no_grad():
        model.eval()
        for images, filenames in test_loader:
            images = images.to(device)
            preds = model(images)

            print(preds.sum().item())

            # if preds.sum().item() != 0:
            #     print(preds.sum())
            # else:
            #     "-"

            # boxes, confidences = decode_preds(preds, conf_thresh=0.1)
            # if boxes.size(0) == 0:
            #     # print(f"No detections for {filenames[0]}")
            #     continue
            #
            # print(f"Got detections for {filenames[0]}")
