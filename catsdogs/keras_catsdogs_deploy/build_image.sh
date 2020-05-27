s2i build . seldonio/seldon-core-s2i-python3:0.13 chaowen/keras_catsdogs_deploy:internal
docker push chaowen/keras_catsdogs_deploy:internal
