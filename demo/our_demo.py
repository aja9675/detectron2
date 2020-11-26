# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import pickle

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances


# Constants
OUTPUT_DIR = "./outputs"
MASK_HEADS = ["MaskRCNNConvUpsampleHead", "MaskRCNNConvUpsampleHeadActivation",
                "MaskRCNNConvUpsampleHeadLoss", "MaskRCNNConvUpsampleHeadBoth" ]

def translate_to_mask(activation, loss):
    mask_head = "MaskRCNNConvUpsampleHead"
    if activation == "sps":
        if loss == "BCE":
            mask_head = "MaskRCNNConvUpsampleHeadActivation"
        elif loss == "reg":
            mask_head = "MaskRCNNConvUpsampleHeadBoth"
        else:
            print("ERROR: unsupported combination of activation and loss")
            print(f"activation = {activation}, loss = {loss}")
            exit(1)
    elif activation == "relu":
        if loss == "BCE":
            mask_head = "MaskRCNNConvUpsampleHead"
        elif loss == "reg":
            mask_head = "MaskRCNNConvUpsampleHeadLoss"
        else:
            print("ERROR: unsupported combination of activation and loss")
            print(f"activation = {activation}, loss = {loss}")
            exit(1)
    else:
        print("ERROR: unsupported combination of activation and loss")
        print(f"activation = {activation}, loss = {loss}")
        exit(1)
    return mask_head


def setup_cfg(args, output, mask_head="MaskRCNNConvUpsampleHead"):
    # fail immediately if a mask head is bad
    if mask_head not in MASK_HEADS:
        print(f"ERROR: bad mask head: {mask_head}")
        exit(1)

    # load config from file and command-line arguments - default config loaded
    cfg = get_cfg()
    #cfg.merge_from_file(args.config_file)
    # load our initial weights
    cfg.merge_from_file(model_zoo.get_config_file(args.config_file))
    cfg.merge_from_list(args.opts)

    # loading other params based on set values

    # set training stuff
    if args.run_type == "train":
        cfg.DATASETS.TRAIN=("person_box_chicken_train",)
        cfg.DATASETS.TEST=()
        cfg.DATALOADER.NUM_WORKERS=4
        cfg.MODEL_WEIGHTS=model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH=4
        cfg.SOLVER.BASE_LR=0.00025
        cfg.SOLVER.MAX_ITER=1400
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE=64
        cfg.MODEL.ROI_HEADS.NUM_CLASSES=3
        cfg.MODEL.ROI_MASK_HEAD.NAME=mask_head
        output_dirname = "output_" + time.strftime("%Y_%m_%d-%H:%M:%S")
        cfg.OUTPUT_DIR=os.path.join(OUTPUT_DIR, output_dirname)
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=False)
        # Save cfg
        cfg_fname = output+"cfg"
        with open(cfg_fname + ".pkl", 'wb') as f:
            pickle.dump(cfg, f, pickle.HIGHEST_PROTOCOL)
    elif args.run_type == "test":
        cfg.MODEL.ROI_MASK_HEAD.NAME=mask_head
    else:
        print("Why am I seeing this, but error: unsupported run type {args.run_type}")
        exit(1)
    #cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Training and Evaluation Demo, with"
            " our changes")
    parser.add_argument(
            "--run_type",
            type=str,
            default="train",
            help="Run Training or Evaluation"
    )
    parser.add_argument(
            "--dataset",
            type=str,
            default="../../datasets/custom_dataset/person_box_chicken_train",
            help="dataset to use - can be for training or evaluation"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        help="Initial weights to use, likely will never change from default of COCO 2017"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="where to save our training results"
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
            "--activation",
            type=str,
            default="relu",
            help="Change activation function being used in mask head branch, default=relu"
    )
    parser.add_argument(
            "--loss",
            type=str,
            default="BCE",
            help="Change loss function used in mask head branch, default=BCE"
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def register_our_dataset():
    """
    Carefully regiseters our dataset for use in the model
    """
    # People, boxes, and chickens
    # These vars will be used later to reference the datasets
    dataset_name_train = "person_box_chicken_train"
    dataset_name_val = "person_box_chicken_val"

    # I'm doing terrible things with hardcoded paths, oh well
    person_box_chicken_train_json = "/home/sean/Documents/school/4th_year/cv/final_project/datasets/custom_dataset/person_box_chicken_train.json"
    person_box_chicken_val_json = "/home/sean/Documents/school/4th_year/cv/final_project/datasets/custom_dataset/person_box_chicken_val.json"
    person_box_chicken_train_image_dir = "/home/sean/Documents/school/4th_year/cv/final_project/datasets/custom_dataset/person_box_chicken_train/"
    person_box_chicken_val_image_dir = "/home/sean/Documents/school/4th_year/cv/final_project/datasets/custom_dataset/person_box_chicken_val/"

    # Careful not to double register, it raises an exception.
    for name in DatasetCatalog.list():
        if name not in [dataset_name_train, dataset_name_val]:
            DatasetCatalog.remove(name)
    if dataset_name_train in DatasetCatalog.list():
      print("Already registered %s dataset" % dataset_name_train)
    else:
      register_coco_instances(dataset_name_train, {}, person_box_chicken_train_json, person_box_chicken_train_image_dir)

    if dataset_name_val in DatasetCatalog.list():
      print("Already registered %s dataset" % dataset_name_val)
    else:
      register_coco_instances(dataset_name_val, {}, person_box_chicken_val_json, person_box_chicken_val_image_dir)


if __name__ == "__main__":
    args = get_parser().parse_args()
    print(args)
    t = time.strftime("%Y_%m_%d-%H:%M:%S")
    output_file = f"./outputs/{args.run_type}_{t}"
    register_our_dataset()
    mask_head = translate_to_mask(args.activation, args.loss)
    print(f"mask head = {mask_head}")
    if args.run_type == "train":
        print("Beginning training run")
        cfg = setup_cfg(args, output_file)
        cfg.MODEL.ROI_MASK_HEAD.NAME=mask_head
        print(f"mask head before trainer = {cfg.MODEL.ROI_MASK_HEAD.NAME}")
        #for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        #    print(dataset_name)
        trainer = DefaultTrainer(cfg)
        print(f"mask head after trainer = {cfg.MODEL.ROI_MASK_HEAD.NAME}")
        trainer.resume_or_load(resume=False)
        trainer.train()
    elif args.run_type == "test":
        print("Beginning evaluation run")
        cfg = setup_cfg(args, output_file, mask_head=mask_head)
        print(cfg)
    else:
        print("Error: unsupported/unexpected run type, aborting...")
        exit(1)





