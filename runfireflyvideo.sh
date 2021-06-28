if [ -e "$1" ]; then

	echo "Video $1 found!"
	videofolder="${1%.mp4}ffmpeg"
	if [ -d "$videofolder" ]; then
		echo "using premade image folder $videofolder"
	elif [ ! -d "$videofolder" ]; then
		echo "creating image folder $videofolder"
		mkdir "$videofolder"
		ffmpeg -i "$1" -vf scale=1280:720,setsar=1:1 -qscale:v 5 "$videofolder/%03d.jpg"
	fi

	build/bin/dso_dataset \
			files="$videofolder" \
			calib=~/dso/firefly/camera.txt \
			preset=0 \
			mode=1

			# files=/home/pootis/dso/firefly/video2ffmpeg \
elif [ ! -e "$1" ]; then
	echo "Video $1 not found!"
fi
