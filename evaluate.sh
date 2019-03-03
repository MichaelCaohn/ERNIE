wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl
./sentencepiece-master/build/src/spm_decode --model sentencepiece.model --input_format piece < output.txt > output_detok.txt
perl multi-bleu-detok.perl newstest2017-ende-ref.de < output_detok.txt
