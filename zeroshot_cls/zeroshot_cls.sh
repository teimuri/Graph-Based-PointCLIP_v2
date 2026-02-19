#!/bin/bash

# Path to dataset
DATASET=scanobjectnn
TARGETDATASET=modelnet40

# Path to dataset
# DATASET=scanobjectnn

TRAINER=PointCLIPV2_ZS
# Trainer configs: rn50, rn101, vit_b32 or vit_b16
CFG=vit_b16

export CUDA_VISIBLE_DEVICES=0
python main.py \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${TARGETDATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${TARGETDATASET} \
--no-train \
--zero-shot \
--gnn-dir output/${TRAINER}/${CFG}/${DATASET}/gnn/my_gnn_model.pth \
--post-search
