#!/bin/sh

python train.py \
	-c config/backbones/mobilenet_v3_large.yml \
	-c config/heads/transformer.yml \
	-c config/misc/augment-yes.yml \
	-c config/data/vietocr-sample-dataset.yml \
	$@
