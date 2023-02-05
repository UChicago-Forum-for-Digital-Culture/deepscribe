#!/bin/bash


DATA_DIR=/local/ecw/DeepScribe_Data_2023-02-04-selected

for PARTITION in train test val
do
    echo $PARTITION
    python deepscribe2/preprocessing/get_hotspots.py --json $DATA_DIR/data_$PARTITION.json\
                                                    --raw_imgs $DATA_DIR/cropped_images \
                                                    --split_imgs $DATA_DIR/all_hotspots/hotspots_$PARTITION 
done
