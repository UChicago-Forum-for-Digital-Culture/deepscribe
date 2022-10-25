#!/bin/bash

DATA_DIR=/local/ecw/data_oct_2022

python deepscribe2/preprocessing/crop_images.py --json $DATA_DIR/imagesWithHotspots.json \
                                                                        --raw_imgs $DATA_DIR/raw_images \
                                                                        --cropped_imgs $DATA_DIR/cropped_images \
                                                                        --cropped_json $DATA_DIR/imagesWithHotspots_cropped.json
