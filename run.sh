export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so:$LD_PRELOAD"
accelerate launch train.py /data /model $1
