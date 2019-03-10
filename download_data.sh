#!/bin/bash
wget https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz
tar xvzf wmt_ende_sp.tar.gz

mv train.en train.de test.en test.de valid.en valid.de data

python preprocess.py -train_src data/train.en -train_tgt data/train.de -valid_src data/valid.en -valid_tgt data/valid.de -save_data data/wmt14 --vocab_size=32000

wget https://s3.amazonaws.com/opennmt-models/transformer-ende-wmt-pyOnmt.tar.gz
tar xvzf transformer-ende-wmt-pyOnmt.tar.gz

./sentencepiece-master/build/src/spm_decode --model sentencepiece.model --input_format piece < "data/valid.de" > "val.txt"
