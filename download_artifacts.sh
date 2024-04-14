#!/bin/bash

# downloads pretrained model artifacts for the classifier and detector. 

ARTIFACTS_DIR=artifacts

mkdir $ARTIFACTS_DIR

wget --directory-prefix=$ARTIFACTS_DIR "https://ochre.lib.uchicago.edu/deepscribe/classifier_epoch=50-step=2091.ckpt"

wget --directory-prefix=$ARTIFACTS_DIR  "https://ochre.lib.uchicago.edu/deepscribe/detector_epoch=358-step=88673.ckpt"