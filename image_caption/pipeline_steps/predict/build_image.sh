#!/bin/bash

docker build . -t chaowen/img_caption_predict:latest
docker push chaowen/img_caption_predict:latest
