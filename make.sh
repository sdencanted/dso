#!/usr/bin/env bash

# grep MemTotal /proc/meminfo
raminkb=`grep MemTotal /proc/meminfo | awk '{print $2}'`
if [ $raminkb -lt 8000000 ]; then
	echo "RAM insufficient! creating swap..."
	mkdir swap # Create a directory to put swapfile
	cd swap
	sudo dd if=/dev/zero of=swapfile bs=1K count=8M # Create swapflie, size = bs * count
	sudo mkswap swapfile                            # Set swapfile
	sudo swapon swapfile                            # Mount
	free -m                                         # View
	cd ..
fi


mkdir build
cd build
cmake ..
make -j4

if [ $raminkb -lt 8000000 ]; then
	#After the install is done
	echo "deactivating swap..."
	cd ..
	cd swap
	sudo swapoff swapfile
	echo "removing swap..."
	cd ..
	sudo rm -rf swap
fi