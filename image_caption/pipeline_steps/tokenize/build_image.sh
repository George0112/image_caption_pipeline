#!/bin/bash

docker build . -t chaowen/img_caption_tokenize:latest
docker push chaowen/img_caption_tokenize:latest