#!/bin/bash

mkdir -p /home/local/ASUAD/yikangli/datasets/mnist

if ! [ -e /home/local/ASUAD/yikangli/datasets/mnist/train-images-idx3-ubyte.gz ]
	then
		wget -P /home/local/ASUAD/yikangli/datasets/mnist/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
fi
gzip -d /home/local/ASUAD/yikangli/datasets/mnist/train-images-idx3-ubyte.gz

if ! [ -e /home/local/ASUAD/yikangli/datasets/mnist/train-labels-idx1-ubyte.gz ]
	then
		wget -P /home/local/ASUAD/yikangli/datasets/mnist/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
fi
gzip -d /home/local/ASUAD/yikangli/datasets/mnist/train-labels-idx1-ubyte.gz

if ! [ -e /home/local/ASUAD/yikangli/datasets/mnist/t10k-images-idx3-ubyte.gz ]
	then
		wget -P /home/local/ASUAD/yikangli/datasets/mnist/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
fi
gzip -d /home/local/ASUAD/yikangli/datasets/mnist/t10k-images-idx3-ubyte.gz

if ! [ -e /home/local/ASUAD/yikangli/datasets/mnist/t10k-labels-idx1-ubyte.gz ]
	then
		wget -P /home/local/ASUAD/yikangli/datasets/mnist/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
fi
gzip -d /home/local/ASUAD/yikangli/datasets/mnist/t10k-labels-idx1-ubyte.gz
