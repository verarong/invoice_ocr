#!/bin/bash

USER=root
TAG=cpu
TF_SERVING_VERSION_GIT_BRANCH="r1.13"
git clone --branch="${TF_SERVING_VERSION_GIT_BRANCH}" https://github.com/tensorflow/serving

TF_SERVING_BUILD_OPTIONS="--copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.1 --copt=-msse4.2"
cd serving && \
  docker build --pull -t $USER/tensorflow-serving-devel:$TAG \
  --build-arg TF_SERVING_VERSION_GIT_BRANCH="${TF_SERVING_VERSION_GIT_BRANCH}" \
  --build-arg TF_SERVING_BUILD_OPTIONS="${TF_SERVING_BUILD_OPTIONS}" \
  -f tensorflow_serving/tools/docker/Dockerfile.devel .

cd serving && \
  docker build -t $USER/tensorflow-serving:$TAG \
  --build-arg TF_SERVING_BUILD_IMAGE=$USER/tensorflow-serving-devel:$TAG \
  -f tensorflow_serving/tools/docker/Dockerfile .