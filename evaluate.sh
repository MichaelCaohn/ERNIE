wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

python translate.py -model /tmp/extra_step_50.pt -src newstest2017-ende-src.en -tgt newstest2017-ende-ref.de -output output.txt -report_time -gpu 1 --from_quantized

./sentencepiece-master/build/src/spm_decode --model sentencepiece.model --input_format piece < output.txt > output_detok.txt

perl multi-bleu-detok.perl newstest2017-ende-ref.de < output_detok.txt
