src=en
prefix=train
main_dir=/users2/hpzhao/gyzhu/wmt17
data_dir=$main_dir/data_10w
model_dir=$main_dir/model_10w
moses_scripts=$main_dir/mosesdecoder/scripts
bpe_scripts=$main_dir/subword-nmt
nematus_home=$main_dir/nematus
# tokenize
cat $data_dir/train.en | \
$moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
    $moses_scripts/tokenizer/tokenizer.perl -a -threads 8 -l $src > $data_dir/$prefix.$src.tok
# truecaser
$moses_scripts/recaser/train-truecaser.perl -corpus $data_dir/$prefix.$src.tok -model $model_dir/truecase-model.$src
$moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$src < $data_dir/$prefix.$src.tok > $data_dir/$prefix.$src.tok.tc
# bpe
python $bpe_scripts/learn_bpe.py -s 100000 < $data_dir/$prefix.$src.tok.tc > $data_dir/$prefix.$src.tok.tc.codes
python $bpe_scripts/apply_bpe.py -c $data_dir/$prefix.$src.tok.tc.codes < $data_dir/$prefix.$src.tok.tc | \
    python $bpe_scripts/get_vocab.py > $data_dir/$prefix.$src.tok.tc.vocab
python $bpe_scripts/apply_bpe.py -c $data_dir/$prefix.$src.tok.tc.codes --vocabulary $data_dir/$prefix.$src.tok.tc.vocab \
    --vocabulary-threshold 20 < $data_dir/$prefix.$src.tok.tc > $data_dir/$prefix.$src.tok.tc.bpe
# build dict
python $bpe_scripts/get_vocab.py < $data_dir/$prefix.$src.tok.tc.bpe > $data_dir/$prefix.$src.tok.tc.bpe.vocab
python $nematus_home/data/build_dictionary.py $data_dir/$prefix.$src.tok.tc.bpe

# dev/test : tokenize
cat $data_dir/valid.en | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
    $moses_scripts/tokenizer/tokenizer.perl -a -threads 8 -l $src > $data_dir/valid.$src.tok
cat $data_dir/test.en | \
    $moses_scripts/tokenizer/normalize-punctuation.perl -l $src | \
    $moses_scripts/tokenizer/tokenizer.perl -a -threads 8 -l $src > $data_dir/test.$src.tok
# dev/test : truecaser
$moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$src < $data_dir/valid.$src.tok > $data_dir/valid.$src.tok.tc
$moses_scripts/recaser/truecase.perl -model $model_dir/truecase-model.$src < $data_dir/test.$src.tok > $data_dir/test.$src.tok.tc
# dev/test : bpe
python $bpe_scripts/apply_bpe.py -c $data_dir/$prefix.$src.tok.tc.codes --vocabulary $data_dir/$prefix.$src.tok.tc.vocab \
    --vocabulary-threshold 20 < $data_dir/valid.$src.tok.tc > $data_dir/valid.$src.tok.tc.bpe
python $bpe_scripts/apply_bpe.py -c $data_dir/$prefix.$src.tok.tc.codes --vocabulary $data_dir/$prefix.$src.tok.tc.vocab \
    --vocabulary-threshold 20 < $data_dir/test.$src.tok.tc > $data_dir/test.$src.tok.tc.bpe
