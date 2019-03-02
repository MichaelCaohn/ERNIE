#!/bin/bash

# download all training data
#wget http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
#wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
#wget http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz

# download dev / test sets
#wget http://www.statmt.org/wmt14/dev.tgz
#wget http://www.statmt.org/wmt14/test-full.tgz

# make temp dir and final data dir
mkdir temp
mkdir wmt14-en-de

# unpack all
tar -xvzf training-parallel-europarl-v7.tgz -C temp
tar -xvzf training-parallel-commoncrawl.tgz -C temp
tar -xvzf training-parallel-nc-v9.tgz -C temp
tar -xvzf dev.tgz -C temp
tar -xvzf test-full.tgz -C temp

# combine en-de files and copy to final data dir
cat temp/commoncrawl.de-en.en temp/training/europarl-v7.de-en.en temp/training/news-commentary-v9.de-en.en > wmt14-en-de/train.en
cat temp/commoncrawl.de-en.de temp/training/europarl-v7.de-en.de temp/training/news-commentary-v9.de-en.de > wmt14-en-de/train.de

cp temp/dev/newstest2013.en wmt14-en-de/val.en
cp temp/dev/newstest2013.de wmt14-en-de/val.de

cp temp/test-full/newstest2014-deen-src.en.sgm wmt14-en-de/test.en
cp temp/test-full/newstest2014-deen-ref.de.sgm wmt14-en-de/test.de

