#!/usr/bin/env bash

# grep MemTotal /proc/meminfo
abspath=$(dirname $(readlink -f "$0"))
raminkb=`grep MemTotal /proc/meminfo | awk '{print $2}'`
if [ $raminkb -lt 8000000 ]; then
	echo "RAM insufficient! creating swap..."
	mkdir $abspath/swap # Create a directory to put swapfile
	cd $abspath/swap
	sudo dd if=/dev/zero of=swapfile bs=1K count=8M # Create swapflie, size = bs * count
	sudo mkswap swapfile                            # Set swapfile
	sudo swapon swapfile                            # Mount
	free -m                                         # View
fi


cd $abspath/..
mkdir build
cd build
cmake ..
make -j4

if [ $raminkb -lt 8000000 ]; then
	#After the install is done
	echo "deactivating swap..."
	cd $abspath/swap
	sudo swapoff swapfile
	echo "removing swap..."
	cd ..
	sudo rm -rf swap
fi