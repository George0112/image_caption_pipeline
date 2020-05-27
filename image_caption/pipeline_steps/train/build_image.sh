#!/bin/bash

docker build . -t chaowen/img_caption_train:latest
docker push chaowen/img_caption_train:latest

