#!/bin/bash

dataset_path=/cluster/home/it_stu176/DataSets/RM-YOLOv5/armor.yaml
config_path=models/yolov5s_rm.yaml
pretrained=weights/yolov5s.pt
pruning_times=100
batch_size=32
project_path=runs/train/yolov5s_rm

# normal train (output to $project_path/exp)
python3 train.py --img 640 --batch $batch_size --epoch 50 \
                 --data $dataset_path --cfg $config_path \
                 --hyp data/hyp.scratch.yaml \
                 --weights $pretrained \
                 --project "$project_path"

# pruning
python3 pruning.py --weights $project_path/exp/weights/last.pt --threshold 1e-3

mv $project_path/exp $project_path/exp1
mkdir $project_path/exp

# repeat pruning for 9 times
for i in $(seq 1 $pruning_times)
do
  # sparse train (output to $project_path/exp$(i+1))
  python3 train.py --img 640 --batch $batch_size --epoch 25 \
                   --data $dataset_path --cfg $config_path \
                   --hyp data/hyp.sparse.yaml \
                   --weights $project_path/exp"$i"/weights/last_pruning.pt \
                   --project "$project_path"

  # pruning (output to $project_path/exp$(i+1))
  python3 pruning.py --weights $project_path/exp"$(expr "$i" + 1)"/weights/last.pt --threshold 1e-3
done

# normal train (output to $project_path/exp10)
python3 train.py --img 640 --batch $batch_size --epoch 25 \
                 --data $dataset_path --cfg $config_path \
                 --hyp data/hyp.scratch.yaml \
                 --weights $project_path/exp"$(expr "$pruning_times" + 1)"/weights/last_pruning.pt \
                 --project "$project_path"
