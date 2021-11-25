#!/bin/bash

path_curr="./"
path_train_src=$path_curr"chr_en_data/train.chr"
path_train_tgt=$path_curr"chr_en_data/train.en"

path_dev_src=$path_curr"chr_en_data/dev.chr"
path_dev_tgt=$path_curr"chr_en_data/dev.en"

path_test_src=$path_curr"chr_en_data/test.chr"
path_test_tgt=$path_curr"chr_en_data/test.en"

path_output=$path_curr"outputs/test_outputs.txt"

path_vocab=$path_curr"vocab.json"

path_model=$path_curr"train/model.bin"

path_run=$path_curr"run.py"
path_run_vocab=$path_curr"vocab.py"

if [ "$1" = "train" ]; then
	# CUDA_VISIBLE_DEVICES=0 python run.py train --save-to=./train/model.bin --train-src=./chr_en_data/train.chr --train-tgt=./chr_en_data/train.en --dev-src=./chr_en_data/dev.chr --dev-tgt=./chr_en_data/dev.en --vocab=vocab.json --cuda --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3
	CUDA_VISIBLE_DEVICES=0 python $path_run train --save-to=$path_model --train-src=$path_train_src --train-tgt=$path_train_tgt --dev-src=$path_dev_src --dev-tgt=$path_dev_tgt --vocab=$path_vocab --cuda --lr=5e-4 --patience=1 --valid-niter=200 --batch-size=32 --dropout=.3
elif [ "$1" = "test" ]; then
	# CUDA_VISIBLE_DEVICES=0 python run.py decode model.bin ./chr_en_data/test.chr ./chr_en_data/test.en outputs/test_outputs.txt --cuda
	CUDA_VISIBLE_DEVICES=0 python $path_run decode $path_model $path_test_src $path_test_tgt $path_output --cuda
elif [ "$1" = "train_local" ]; then
	# python run.py train --train-src=./chr_en_data/train.chr --train-tgt=./chr_en_data/train.en --dev-src=./chr_en_data/dev.chr --dev-tgt=./chr_en_data/dev.en --vocab=vocab.json --lr=5e-4
	python $path_run train --train-src=$path_train_src --train-tgt=$path_train_tgt --dev-src=$path_dev_src --dev-tgt=$path_dev_tgt --vocab=$path_vocab --lr=5e-4
elif [ "$1" = "test_local" ]; then
    # python run.py decode model.bin ./chr_en_data/test.chr ./chr_en_data/test.en outputs/test_outputs.txt
	python $path_run decode $path_model $path_test_src $path_test_tgt $path_output
elif [ "$1" = "vocab" ]; then
	# python vocab.py --train-src=./chr_en_data/train.chr --train-tgt=./chr_en_data/train.en vocab.json
	python $path_run_vocab --train-src=$path_train_src --train-tgt=$path_train_tgt $path_vocab		
else
	echo "Invalid Option Selected"
fi
