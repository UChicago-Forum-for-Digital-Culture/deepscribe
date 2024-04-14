#!/bin/bash

DATA_FOLDER=data

mkdir $DATA_FOLDER

cd $DATA_FOLDER && wget  "https://ochre.lib.uchicago.edu/deepscribe/deepscribe_2023_02_04_public.tar.gz" && tar -xzf deepscribe_2023_02_04_public.tar.gz