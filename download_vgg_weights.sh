#!/usr/bin/env bash
# make the directory to put the vgg19 pre-trained model
mkdir vgg19/
cd vgg19/
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
sudo tar xvf ./vgg_19_2016_08_28.tar.gz