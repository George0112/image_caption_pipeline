#!/bin/bash

docker build . -t chaowen/img_caption_preprocess:latest
docker push chaowen/img_caption_preprocess:latest
