#!/bin/bash

DATA_DIR=/local/ecw/data_oct_2022

python /local/ecw/deepscribe2/deepscribe2/preprocessing/split_tablet.py --json $DATA_DIR/imagesWithHotspots_cropped.json \
                                                                        --splits 0.8 0.1 0.1 \
                                                                        --fold_suffixes train val test \
                                                                        --prefix $DATA_DIR/data_