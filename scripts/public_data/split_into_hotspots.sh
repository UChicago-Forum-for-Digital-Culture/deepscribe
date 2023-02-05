#!/bin/bash


DATA_DIR=/local/ecw/DeepScribe_Data_2023-02-04-selected/public

for PARTITION in train test val
do
    echo $PARTITION
    python deepscribe2/preprocessing/get_hotspots.py --json $DATA_DIR/data_public_$PARTITION.json\
                                                    --raw_imgs $DATA_DIR/cropped_images_public \
                                                    --split_imgs $DATA_DIR/public_hotspots/hotspots_$PARTITION 
done
