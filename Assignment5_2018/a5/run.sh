#!/bin/bash

path_curr="/content/drive/MyDrive/speech_research/lecture/a5_2018/a5"

path_train_src=$path_curr"/en_es_data/train.es"
path_train_tgt=$path_curr"/en_es_data/train.en"

path_dev_src=$path_curr"/en_es_data/dev.es"
path_dev_tgt=$path_curr"/en_es_data/dev.en"

path_vocab=$path_curr"/vocab.json"

path_test_src=$path_curr"/en_es_data/test.es"
path_test_tgt=$path_curr"/en_es_data/test.en"

path_output=$path_curr"outputs/test_outputs.txt"

path_model=$path_curr"/model.bin"

path_run=$path_curr"/run.py"

if [ "$1" = "train" ]; then
	CUDA_VISIBLE_DEVICES=0 python $path_run train --save-to=$path_model --train-src=$path_train_src --train-tgt=$path_train_tgt \
        --dev-src=$path_dev_src --dev-tgt=$path_dev_tgt --vocab=$path_vocab --cuda
elif [ "$1" = "test" ]; then
    mkdir -p $path_output
    touch $path_output
    CUDA_VISIBLE_DEVICES=0 python $path_run decode $path_model $path_test_src $path_test_tgt $path_output --cuda
elif [ "$1" = "train_local_q1" ]; then
	python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q1.json --batch-size=2 \
        --valid-niter=100 --max-epoch=101 --no-char-decoder
elif [ "$1" = "test_local_q1" ]; then
    mkdir -p outputs
    touch outputs/test_local_outputs.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q1.txt \
        --no-char-decoder
elif [ "$1" = "train_local_q2" ]; then
	python run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q2.json --batch-size=2 \
        --max-epoch=201 --valid-niter=100
elif [ "$1" = "test_local_q2" ]; then
    mkdir -p outputs
    touch outputs/test_local_outputs.txt
    python run.py decode model.bin ./en_es_data/test_tiny.es ./en_es_data/test_tiny.en outputs/test_outputs_local_q2.txt 
elif [ "$1" = "vocab" ]; then
    python vocab.py --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        --size=200 --freq-cutoff=1 vocab_tiny_q1.json
    python vocab.py --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
        vocab_tiny_q2.json
	python vocab.py --train-src=./en_es_data/train.es --train-tgt=./en_es_data/train.en vocab.json
else
	echo "Invalid Option Selected"
fi
