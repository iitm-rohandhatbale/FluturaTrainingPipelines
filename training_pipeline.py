import os
import yaml
import pprint
import json
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
from mlflow.tracking  import MlflowClient

class TrainPytorchYOLOv5:
    def __init__(self):
        self.PATH_TO_PROJECTS = r'.'
        self.PATH_TO_MLFLOW_DIR ='.'

        self.system_dict = {
            'project_params': {},
            'system_params': {},
            'dataset_params': {},
            'model_params': {},
            'hyper_params': {},
            'training_params': {}
        }

    def create_data_yaml(self, dataset_path, labels, val_data=False, test_data=False):
        """
        creates data.yaml file
        Args:
            test_data: (bool) whether test data present or not
            val_data:  (bool) whether val data present or not
            dataset_path: path to dataset directory
            labels: list of labels

        Returns:
            creates data.yaml file
        """
        val_path = os.path.join(dataset_path, 'images', 'val') if val_data else ''
        test_path = os.path.join(dataset_path, 'images', 'test') if test_data else ''
        data = {
            'path': dataset_path,
            'train': os.path.join(dataset_path, 'images', 'train'),
            'val': val_path,
            'test': test_path,
            'nc': len(labels),
            'names': labels
        }
        try:
            os.makedirs(os.path.join(self.PATH_TO_PROJECTS,
                                     self.system_dict['project_params']['project_name'],
                                     self.system_dict['project_params']['experiment_name']))
        except:
            pass

        path_to_save_data_yaml = os.path.join(self.PATH_TO_PROJECTS,
                                              self.system_dict['project_params']['project_name'],
                                              self.system_dict['project_params']['experiment_name'],
                                              'data.yaml')

        with open(path_to_save_data_yaml, 'w') as file:
            yaml.dump(data, file)

        return path_to_save_data_yaml

    def set_project_params(self, project_name, experiment_name, already_exist=0):
        """
        function to set project related parameters
        Args:
            project_name: Name of the project
            experiment_name: Name of the Experiment
            already_exist: (bool) Whether the project or experiment already exist or not

        Returns:
            Sets project parameters to system dict

        """
        assert project_name is not None and experiment_name is not None, 'Project Name and Experiment Name cannot be None'

        self.system_dict['project_params']['project_name'] = project_name
        self.system_dict['project_params']['experiment_name'] = experiment_name
        self.system_dict['project_params']['already_exist'] = already_exist

        # pprint.pprint(self.system_dict['project_params'])

    def set_system_params(self, device='', workers=8):
        """
        Set dataset params to system dict
        Args:
            device: cuda device, i.e. 0 or 0,1,2,3 or cpu (defalut: '')
            workers:max dataloader workers (per RANK in DDP mode) (default:8)
        """
        self.system_dict['system_params']['device'] = device
        self.system_dict['system_params']['workers'] = workers

        # pprint.pprint(self.system_dict['system_params'])

    def set_dataset_params(self, dataset_path, labels, val_data=False, test_data=False):
        """
        Sets dataset parameters to system dict
        Args:
            dataset_path: path to dataset folder
            labels: list of labels
            val_data: (bool) whether val data present or not
            test_data: (bool) whether test data present or not

        """
        assert dataset_path is not None and labels is not None, 'Provide dataset path and list of labels'
        data_yaml_path = self.create_data_yaml(dataset_path, labels, val_data, test_data)

        self.system_dict['dataset_params']['dataset_path'] = dataset_path
        self.system_dict['dataset_params']['labels'] = labels
        self.system_dict['dataset_params']['val_data'] = val_data
        self.system_dict['dataset_params']['test_data'] = test_data
        self.system_dict['dataset_params']['data_yaml_path'] = data_yaml_path

        # pprint.pprint(self.system_dict['dataset_params'])

    def set_model_params(self, model_type='nano', pretrained=True, cfg_path=None, resume_training=False,
                         prev_project_name=None, prev_experiment_name=None, weight_path = None):
        """
        set model parameters
        Args:
            model_type: type of model to train. ['nano', 'small', 'medium', 'large', 'xlarge']
            pretrained: (bool) whether to use pretrained model weight or not
            cfg_path: if not pretrained then have to provode cfg path to  initialize model training from random weight initialization
            resume_training: (bool) whether to resume training
            weight_path: if resume training then provide path  to weight file
            prev_project_name: name of the project from which model training is resumed
            prev_experiment_name: name of the experiment from whichmodel training is resumed


        """
        # assert (pretrained is False and cfg_path is None) or weight_path is not None, 'If not pretrained then please provide config file for ' \
        #                                         'selected model type or provide path to weight file'
        # print("resume_training: ",resume_training, type(resume_training))
        # assert resume_training is False and prev_project_name is None and prev_experiment_name is None, 'If Resume training then provide path to pretrained weight'

        if (pretrained==False) and (resume_training==False):
            weight_path = weight_path
        elif resume_training:
            weight_path = os.path.join(self.PATH_TO_PROJECTS, prev_project_name, prev_experiment_name, 'weights',
                                       'last.pt')
        elif pretrained and model_type == 'nano':
            weight_path = 'yolov5n.pt'
        elif pretrained and model_type == 'small':
            weight_path = 'yolov5s.pt'
        elif pretrained and model_type == 'medium':
            weight_path = 'yolov5m.pt'
        elif pretrained and model_type == 'large':
            weight_path = 'yolov5l.pt'
        elif pretrained and model_type == 'xlarge':
            weight_path = 'yolov5x.pt'

        if cfg_path is None:
            cfg_path = """''"""
        self.system_dict['model_params']['model_type'] = model_type
        self.system_dict['model_params']['pretrained'] = pretrained
        self.system_dict['model_params']['cfg_path'] = cfg_path
        self.system_dict['model_params']['resume_training'] = resume_training
        self.system_dict['model_params']['weight_path'] = weight_path

        # pprint.pprint(self.system_dict['model_params'])

    def set_hyper_params(self, epochs=30, batch_size=1, img_size=640, optimizer='SGD', label_smoothing=0.0):
        """
        Sets hyperparameters to system dict
        Args:
            epochs: No of epoch to train the model (default: 30)
            batch_size: Batch size to train the model (default:1)
            img_size: input image size (drfault: 640)
            optimizer: optimizer from 'SGD', 'Adam', 'AdamW'. (default: 'SGD')
            label_smoothing: Label smoothing epsilon (default: 0.0)
        """

        self.system_dict['hyper_params']['epochs'] = epochs
        self.system_dict['hyper_params']['batch_size'] = batch_size
        self.system_dict['hyper_params']['img_size'] = img_size
        self.system_dict['hyper_params']['optimizer'] = optimizer
        self.system_dict['hyper_params']['label_smoothing'] = label_smoothing

        # pprint.pprint(self.system_dict['hyper_params'])

    def set_training_params(self, patience=10, save_period=-1, freeze=None):
        """
        sets training params to system dict
        Args:
            patience: EarlyStopping patience (epochs without improvement) (default: 10)
            save_period:Save checkpoint every x epochs (disabled if < 1) (default: -1)
            freeze:Freeze layers: backbone=10, first3=0 1 2 (default: [0])
        """
        if freeze is None:
            freeze = 0

        self.system_dict['training_params']['patience'] = patience
        self.system_dict['training_params']['save_period'] = save_period
        self.system_dict['training_params']['freeze'] = freeze

        # pprint.pprint(self.system_dict['training_params'])

    def log_to_mlflow(self):
        project_name = self.system_dict['project_params']['project_name']
        experiment_name = self.system_dict['project_params']['experiment_name']

        assert os.path.exists(os.path.join(self.PATH_TO_PROJECTS, project_name, experiment_name)), "Nothing to log to Mlflow"

        EXPERIMENT_NAME = project_name
        RUN_NAME = experiment_name

        mlflow.set_tracking_uri(f"file:///{self.PATH_TO_MLFLOW_DIR}")
        artifact_repository = os.path.join(self.PATH_TO_PROJECTS, EXPERIMENT_NAME, RUN_NAME)

        client = MlflowClient()

        try:
            experiment_id = client.create_experiment(EXPERIMENT_NAME)
        except:
            experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

        with mlflow.start_run(experiment_id=experiment_id, run_name=RUN_NAME) as run:
            run_id = run.info.run_uuid

            client.set_tag(run_id, "mlflow.note.content", f"{EXPERIMENT_NAME, RUN_NAME}")

            # mlflow.log_params(self.system_dict)
            mlflow.log_params(self.system_dict['system_params'])
            mlflow.log_params(self.system_dict['dataset_params'])
            mlflow.log_params(self.system_dict['model_params'])
            mlflow.log_params(self.system_dict['hyper_params'])
            mlflow.log_params(self.system_dict['training_params'])


            df = pd.read_csv(os.path.join(self.PATH_TO_PROJECTS, project_name, experiment_name, 'results.csv'))
            new_columns = [i.strip() for i in list(df.columns)]
            df.columns = new_columns

            train_logs = {
                'total_epochs': df['epoch'][len(df) - 1] + 1,
                'train_box_loss': df['train/box_loss'][len(df) - 1],
                'train_obj_loss': df['train/obj_loss'][len(df) - 1],
                'train_cls_loss': df['train/cls_loss'][len(df) - 1],
                'train_total_loss': df['train/box_loss'][len(df) - 1] + df['train/obj_loss'][len(df) - 1] +
                                    df['train/cls_loss'][len(df) - 1]
            }

            val_logs = {
                'val_box_loss': df['val/box_loss'][len(df) - 1],
                'val_obj_loss': df['val/obj_loss'][len(df) - 1],
                'val_cls_loss': df['val/cls_loss'][len(df) - 1],
                'val_total_loss': df['val/box_loss'][len(df) - 1] + df['val/obj_loss'][len(df) - 1] +
                                  df['val/cls_loss'][len(df) - 1]
            }

            metrics = {
                'precision': df['metrics/precision'][len(df) - 1],
                'recall': df['metrics/recall'][len(df) - 1],
                'mAP_0.5': df['metrics/mAP_0.5'][len(df) - 1],
                'metrics/mAP_0.5_0.95': df['metrics/mAP_0.5:0.95'][len(df) - 1],
            }

            mlflow.log_metrics(train_logs)
            mlflow.log_metrics(val_logs)
            mlflow.log_metrics(metrics)

            def plot_loss_curve(df):
                fig = plt.figure(figsize = (10, 10))
                plt.plot(df.epoch, df['train/box_loss'] + df['train/obj_loss'] + df['train/cls_loss'], color='red',
                         linewidth=2, marker='o', label='train_loss')
                plt.plot(df.epoch, df['val/box_loss'] + df['val/obj_loss'] + df['val/cls_loss'], color='green',
                         linewidth=2, marker='o', label='val_loss')
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title("Train vs val loss curve")
                plt.legend()
                return fig

            loss_curve = plot_loss_curve(df)
            # map_curve = plot_map_curve(eval_logs, labels=labels)

            mlflow.log_figure(loss_curve, "loss_curve.png")

    def train(self):
        train_command = f"python yolov5//train.py " \
                        f"--weights {self.system_dict['model_params']['weight_path']} " \
                        f"--data {self.system_dict['dataset_params']['data_yaml_path']} " \
                        f"--epochs {self.system_dict['hyper_params']['epochs']} " \
                        f"--batch-size {self.system_dict['hyper_params']['batch_size']} " \
                        f"--img-size {self.system_dict['hyper_params']['img_size']} " \
                        f"--device {self.system_dict['system_params']['device']} " \
                        f"--optimizer {self.system_dict['hyper_params']['optimizer']} " \
                        f"--workers {self.system_dict['system_params']['workers']} " \
                        f"--project {os.path.join(self.PATH_TO_PROJECTS, self.system_dict['project_params']['project_name'])} " \
                        f"--name {self.system_dict['project_params']['experiment_name']} " \
                        f"--exist-ok " \
                        f"--label-smoothing {self.system_dict['hyper_params']['label_smoothing']} " \
                        f"--patience {self.system_dict['training_params']['patience']} " \
                        f"--freeze {self.system_dict['training_params']['freeze']} " \
                        f"--save-period {self.system_dict['training_params']['save_period']}"
        print(train_command)
        return train_command
        # os.system(train_command)


if __name__ == "__main__":
    with open(r"D:\PyQt_UI\UI\json_collection\tempconfig\pytorch_yolov5_config.json", 'r') as file:
        data = json.load(file)

    gtf = TrainPytorchYOLOv5()
    # print(gtf.set_project_params.__doc__)
    # gtf.set_project_params(project_name="Project-11",
    # experiment_name="Experiment-28", already_exist=0)
    gtf.set_project_params(project_name=data["project_params"]["project_name"], experiment_name=data["project_params"]["experiment_name"], already_exist=0)
    gtf.set_system_params(device='cpu', workers=8)


    gtf.set_dataset_params(dataset_path=data["dataset_params"]["dataset_path"],
                           labels=data["dataset_params"]["labels"],
                           val_data = bool(data["dataset_params"]["val_data"]),
                           test_data = bool(data["dataset_params"]["test_data"]))

    gtf.set_model_params(model_type=data["model_params"]["model_type"],
                         pretrained=bool(data["model_params"]["pretrained"]),
                         cfg_path=None,
                         resume_training=bool(data["model_params"]["resume_training"]),
                         prev_project_name=data["model_params"]["prev_project_name"],
                         prev_experiment_name=data["model_params"]["prev_experiment_name"],
                         weight_path=data["model_params"]["weight_path"])

    gtf.set_hyper_params(epochs=data["hyper_params"]["epochs"],
                         batch_size=data["hyper_params"]["batch_size"],
                         img_size=data["hyper_params"]["img_size"],
                         optimizer=data["hyper_params"]["optimizer"],
                         label_smoothing=0.0)

    gtf.set_training_params(patience=data["training_params"]["patience"],
                            save_period=data["training_params"]["save_period"],
                            freeze=None)
    train_command = gtf.train()

    os.system(train_command)


    # gtf = TrainPytorchTOLOv5()

    # # print(gtf.set_project_params.__doc__)
    # gtf.set_project_params(project_name="Project-11", experiment_name="Experiment-22", already_exist=0)
    # gtf.set_system_params(device='cpu', workers=8)
    # gtf.set_dataset_params(dataset_path=r'D:\MLFlow\YOLOv5\coco128',
    #                        labels=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    #                                'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    #                                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    #                                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    #                                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    #                                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    #                                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    #                                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    #                                'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
    # gtf.set_model_params(model_type='nano', pretrained=False, cfg_path=None, resume_training=False,
    #                      prev_project_name=None, prev_experiment_name=None, weight_path = r"E:\pellet detection\yolov5Pellet\weights\best.pt")
    # gtf.set_hyper_params(epochs=3, batch_size=1, img_size=512, optimizer='SGD', label_smoothing=0.0)
    # gtf.set_training_params()
    # gtf.train()




    # # try:
    # #     gtf.train()
    # # finally:
    # #     gtf.log_to_mlflow()
