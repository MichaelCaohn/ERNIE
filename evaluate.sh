wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

python translate.py -model /tmp/extra_step_100.pt -src newstest2017-ende-src.en -tgt newstest2017-ende-ref.de -output output.txt -report_time -gpu 0

./sentencepiece-master/build/src/spm_decode --model sentencepiece.model --input_format piece < output.txt > output_detok.txt

perl multi-bleu-detok.perl newstest2017-ende-ref.de < output_detok.txt
