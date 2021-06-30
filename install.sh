#!/usr/bin/env bash
sudo apt update
sudo apt install -y libsuitesparse-dev libeigen3-dev libboost-all-dev
sudo apt install -y libopencv-dev ffmpeg
sudo apt install -y cmake git  libglu1-mesa-dev freeglut3-dev mesa-common-dev glew-utils libglew-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff5-dev libopenexr-dev libpcl-dev pcl-tools

cd ~
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ..
cmake --build .
cd ~
# git clone https://github.com/sdencanted/dso

cd dso
./make.sh