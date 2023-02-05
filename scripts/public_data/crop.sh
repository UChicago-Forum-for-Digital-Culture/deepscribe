#!/bin/bash

DATA_DIR=/local/ecw/DeepScribe_Data_2023-02-04-selected

python deepscribe2/preprocessing/crop_images.py --json $DATA_DIR/imagesWithHotspots_public.json \
                                                --raw_imgs $DATA_DIR/images_public \
                                                --cropped_imgs $DATA_DIR/cropped_images_public \
                                                --cropped_json $DATA_DIR/imagesWithHotspots_cropped_public.json
