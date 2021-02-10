#!/bin/bash

dataset_path=/cluster/home/it_stu176/DataSets/RM-YOLOv5/armor.yaml
config_path=models/yolov5s_rm.yaml
pruning_times=9

# normal train (output to runs/train/exp)
python3 train.py --img 640 --batch 32 --epoch 50 \
                 --data $dataset_path --cfg $config_path \
                 --hyp data/hyp.scratch.yaml \
                 --weights weights/yolov5s.pt

# pruning
python3 pruning.py --weights runs/train/exp/weights/last.pt --threshold 1e-3

mv runs/train/exp runs/train/exp1
mkdir runs/train/exp

# repeat pruning for 9 times
for i in $(seq 1 $pruning_times)
do
  # sparse train (output to runs/train/exp$(i+1))
  python3 train.py --img 640 --batch 32 --epoch 25 \
                   --data $dataset_path --cfg $config_path \
                   --hyp data/hyp.sparse.yaml \
                   --weights runs/train/exp"$i"/weights/last_pruning.pt

  # pruning (output to runs/train/exp$(i+1))
  python3 pruning.py --weights runs/train/exp"$(expr "$i" + 1)"/weights/last.pt --threshold 1e-3
done

# normal train (output to runs/train/exp10)
python3 train.py --img 640 --batch 32 --epoch 25 \
                 --data $dataset_path --cfg $config_path \
                 --hyp data/hyp.scratch.yaml \
                 --weights runs/train/exp"$(expr "$pruning_times" + 1)"/weights/last_pruning.pt
