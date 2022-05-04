import torch
import PIL
import cv2
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm
import numpy as np
import argparse
import json


class InferPytorchYOLOv5:
    def __init__(self):
        self.PATH_TO_PROJECTS = r'D:\PyQt_UI\UI\projects'
        self.PATH_TO_MLFLOW_DIR = 'D:/PyQt_UI/mlflow_experiments/'


    def load_model(self, path_to_yolov5_dir, path_to_model, iou, confidence_threshold):
        start = time.time()
        model = torch.hub.load(path_to_yolov5_dir, 'custom', path=path_to_model, source='local')

        # setting model Configuration
        model.iou = iou
        model.conf = confidence_threshold
        load_time = time.time() - start
        return model, load_time

    def infer_on_single_img(self, img_path, input_size, model):
        cv_img = cv2.imread(img_path)
        result_img = cv_img.copy()
        start = time.time()
        res = model(cv_img, size=input_size)
        lis = res.xyxy[0].tolist()

        infer_time = time.time() - start
        detection_dict = {"boxes": [], "scores": [], "labels": []}
        for l in lis:
            detection_dict["boxes"].append(l[0:4])
            detection_dict["scores"].append(l[4])
            detection_dict["labels"].append(l[5])

            x1, y1, x2, y2 = [int(i) for i in l[0:4]]
            result_img = cv2.rectangle(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            result_img = cv2.putText(result_img, f"{l[5]}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

        return result_img, detection_dict, infer_time


if __name__ == "__main__":
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--weight_file", required=True, default="best.pt",
    # 				help="name of the weight file")
    # ap.add_argument("--test_images_flag", required=True, default=True, help="whether input is test images or test dataset")
    # ap.add_argument("--test_dataset_flag", required=True, default=False, help="whether input is test images or test dataset")
    # ap.add_argument("--fnames", help="list of image paths")
    # ap.add_argument("--dname", help="path to test dir")
    # ap.add_argument("--conf_thresh", help="Detection confidence threshold")
    # ap.add_argument("--iou_thresh", help="Detection iou threshold for nms")

    # args = vars(ap.parse_args())

    obj = InferPytorchYOLOv5()

    with open(r"D:\PyQt_UI\UI\json_collection\tempconfig\test_input_params.json", "r") as file:
        data = json.load(file)

    with open(r"D:\PyQt_UI\UI\json_collection\tempconfig\pytorch_yolov5_config.json", 'r') as file:
        train_config = json.load(file)

    if data['test_images_flag']:
        fnames = data['fnames']
        project_name = data['project_name']
        experiment_name = data['experiment_name']
        weight_file = data['weight_file']
        iou_thresh = data['iou_thresh']
        conf_thresh = data['conf_thresh']

        img_size = train_config['hyper_params']['img_size']

        path_to_model = os.path.join(obj.PATH_TO_PROJECTS, project_name, experiment_name, "weights", weight_file)

        model, load_time = obj.load_model(path_to_yolov5_dir=r'D:\MLFlow\YOLOv5\yolov5',
                                          path_to_model=path_to_model,
                                          iou=iou_thresh,
                                          confidence_threshold=conf_thresh)
        print(f"model_loaded in {load_time} seconds")

        result = {}
        for img_path in fnames:
            img_name = os.path.basename(img_path)
            result[img_name] = {}
            output_img, detection_dict, infer_time = obj.infer_on_single_img(
                img_path=img_path,
                input_size=img_size,
                model=model
            )
            print("infer_time: ", infer_time)
            result[img_name]['detections'] = detection_dict
            # result[img_name]['output_img'] = output_img.to_list()
            result[img_name]['infer_time'] = infer_time

        data['test_result'] = result

        with open(r"D:\PyQt_UI\UI\json_collection\tempconfig\test_input_params.json", "w") as file:
            json.dump(data, file)
