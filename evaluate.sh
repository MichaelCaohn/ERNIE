
wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

output=$1
test_fn=$2
out_detok=$output"_detok"
echo $out_detok
./sentencepiece-master/build/src/spm_decode --model sentencepiece.model --input_format piece < $output > $out_detok
perl multi-bleu-detok.perl $2 < $out_detok
