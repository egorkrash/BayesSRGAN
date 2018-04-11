#!/usr/bin/env bash

# setup requirements
# I don't know why but upgrade is needed before installing reqs
    sudo pip install --upgrade tensorflow-gpu
sudo pip install -r requirements.txt

# download data
# comment following three lines if you want to use your data set

wget https://www.dropbox.com/s/oif0774okcm7x7z/dataset.zip
unzip -qq dataset.zip
rm -rf dataset.zip


# download VGG weights

# make the directory to put the vgg19 pre-trained model
mkdir vgg19/
cd vgg19/
wget http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
sudo tar xvf ./vgg_19_2016_08_28.tar.gz
sudo chmod 777 vgg_19.ckpt
cd ..

# make checkpoint directory for training
mkdir checkpoint/

# download BSRGAN weights for default setting
cd checkpoint/
wget https://www.dropbox.com/s/rqb9px5abjvueu6/weights_10000.npz