#!/bin/bash

DATA_DIR=/local/ecw/DeepScribe_Data_2023-02-04-selected

python deepscribe2/preprocessing/split_tablet.py --json $DATA_DIR/imagesWithHotspots_cropped_public.json \
                                                                        --splits 0.8 0.1 0.1 \
                                                                        --fold_suffixes train val test \
                                                                        --prefix $DATA_DIR/data_public