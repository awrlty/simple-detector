import os
import shutil

import config


def integrate_labels():  # integrate multiple label files into one text file (yolo data format)
    from_dir = r"D:\Projects\2023_daqs_exterior_wall_quality_inspector\dataset\class_labeling\opening_corner\labels_yolo_format"
    new_txt = r"D:\Projects\2023_daqs_exterior_wall_quality_inspector\dataset\class_labeling\opening_corner\labels_integrated.txt"

    label_files = [file for file in os.listdir(from_dir) if file.lower().endswith(".txt")]
    for file in label_files:
        with open(os.path.join(from_dir, file), "r") as f:
            points = f.readlines()
        if not len(points):
            continue

        filename = file.split(".")[0]
        try:
            points = [list(map(float, pt.strip().split(" "))) for pt in points]
        except ValueError:
            with open(new_txt, "a+") as f:
                f.write(f"{filename}.jpg\n")
            continue

        bboxes = ""
        for pt in points:  # margin 10px for each direction
            x, y = pt[0], pt[1]
            w, h = 40/1024, 40/1024
            bboxes += f"{x} {y} {w} {h} 0 "

        with open(new_txt, "a+") as f:
            f.write(f"{filename}.jpg {bboxes}\n")


def convert_label_format():  # from yolo to voc
    from_dir = r"D:\Projects\2023_daqs_exterior_wall_quality_inspector\dataset\class_labeling\opening_corner\labels_yolo_format"
    new_txt = r"D:\Projects\2023_daqs_exterior_wall_quality_inspector\dataset\class_labeling\opening_corner\labels_voc_format_large.txt"

    label_files = [file for file in os.listdir(from_dir) if file.lower().endswith(".txt")]
    for file in label_files:
        with open(os.path.join(from_dir, file), "r") as f:
            points = f.readlines()
        if not len(points):
            continue

        filename = file.split(".")[0]
        try:
            points = [list(map(float, pt.strip().split(" "))) for pt in points]
        except ValueError:
            with open(new_txt, "a+") as f:
                f.write(f"{filename}.jpg\n")
            continue

        bboxes = ""
        for pt in points:  # margin 10px for each direction
            x1 = max(int(pt[0] * config.IMAGE_SIZE) - 20, 0)
            y1 = max(int(pt[1] * config.IMAGE_SIZE) - 20, 0)
            x2 = min(int(pt[0] * config.IMAGE_SIZE) + 20, config.IMAGE_SIZE)
            y2 = min(int(pt[1] * config.IMAGE_SIZE) + 20, config.IMAGE_SIZE)
            bboxes += f"{x1} {y1} {x2} {y2} 0 "

        with open(new_txt, "a+") as f:
            f.write(f"{filename}.jpg {bboxes}\n")


if __name__ == "__main__":
    # convert_label_format()
    integrate_labels()
