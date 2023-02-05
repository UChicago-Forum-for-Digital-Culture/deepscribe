#!/bin/bash

DATA_DIR=/local/ecw/DeepScribe_Data_2023-02-04-selected

python deepscribe2/preprocessing/crop_images.py --json $DATA_DIR/imagesWithHotspots.json \
                                                --raw_imgs $DATA_DIR/images \
                                                --cropped_imgs $DATA_DIR/cropped_images \
                                                --cropped_json $DATA_DIR/imagesWithHotspots_cropped.json
