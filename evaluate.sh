
wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

output=$1
out_detok=$output"_detok"
echo $out_detok
./sentencepiece-master/build/src/spm_decode --model sentencepiece.model --input_format piece < $output > $out_detok
perl multi-bleu-detok.perl val.txt < $out_detok
