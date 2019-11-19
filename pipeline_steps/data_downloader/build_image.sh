#!/bin/bash

docker build . -t chaowen/img_caption_data_downloader:latest
docker push chaowen/img_caption_data_downloader:latest