#!/bin/bash

cd pipeline_steps/data_downloader && sh build_image.sh &&
cd ../preprocess && sh build_image.sh &&
cd ../tokenize && sh build_image.sh &&
cd ../train && sh build_image.sh &&
cd ../predict && sh build_image.sh