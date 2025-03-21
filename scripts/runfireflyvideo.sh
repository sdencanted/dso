#!/usr/bin/env bash
abspath=$(dirname $(readlink -f "$0"))
cd $abspath
if [ -e "$1" ]; then
	echo "Video $1 found!"
	cd ..
	mkdir save
	cd $abspath

	videofolder="${1%.mp4}ffmpeg"
	if [ -d "$videofolder" ]; then
		echo "using premade image folder $videofolder"
	elif [ ! -d "$videofolder" ]; then
		echo "creating image folder $videofolder"
		mkdir "$videofolder"
		ffmpeg -i "$1" -vf scale=1280:720,setsar=1:1 -qscale:v 5 "$videofolder/%03d.jpg"
	fi
	fps=`ffmpeg -i "$1" 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p"`
	echo $fps
	echo $abspath/fireflycamera.txt
	../build/bin/dso_dataset \
			files="$videofolder" \
			calib=$abspath/fireflycamera.txt \
			preset=0 \
			mode=1 \
			quiet=1 \
			fps=$fps
elif [ ! -e "$1" ]; then
	echo "Video $1 not found!"
fi
# read -p "Press any key to resume ..."