#!/usr/bin/env bash

cd $(dirname $0)

# grep MemTotal /proc/meminfo
raminkb=`grep MemTotal /proc/meminfo | awk '{print $2}'`
if [ $raminkb \< 8000000]; then
	echo "RAM insufficient! creating swap..."]
	mkdir swap # Create a directory to put swapfile
	cd swap
	sudo dd if=/dev/zero of=swapfile bs=1M count=2k # Create swapflie, size = bs * count
	sudo mkswap swapfile                            # Set swapfile
	sudo swapon swapfile                            # Mount
	free -m                                         # View
fi



mkdir build
cd build
cmake ..
make -j4

if [ $raminkb \< 8000000]; then
	echo "removing swap..."
	#After the install is done
	cd $(dirname $0)
	cd swap
	sudo swapoff swapfile
	cd $(dirname $0)
	sudo rm -rf swap
fi