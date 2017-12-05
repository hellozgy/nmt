tgt=zh
prefix=train
main_dir=/users2/hpzhao/gyzhu/wmt17
data_dir=$main_dir/data_10w
model_dir=$main_dir/model_10w
moses_scripts=$main_dir/mosesdecoder/scripts
bpe_scripts=$main_dir/subword-nmt
nematus_home=$main_dir/nematus
# bpe
python $bpe_scripts/learn_bpe.py -s 120000 < $data_dir/$prefix.$tgt.tok > $data_dir/$prefix.$tgt.tok.codes
python $bpe_scripts/apply_bpe.py -c $data_dir/$prefix.$tgt.tok.codes < $data_dir/$prefix.$tgt.tok | \
    python $bpe_scripts/get_vocab.py > $data_dir/$prefix.$tgt.tok.vocab
python $bpe_scripts/apply_bpe.py -c $data_dir/$prefix.$tgt.tok.codes --vocabulary $data_dir/$prefix.$tgt.tok.vocab \
    --vocabulary-threshold 20 < $data_dir/$prefix.$tgt.tok > $data_dir/$prefix.$tgt.tok.bpe
# build dict
python $bpe_scripts/get_vocab.py < $data_dir/$prefix.$tgt.tok.bpe > $data_dir/$prefix.$tgt.tok.bpe.vocab
python $nematus_home/data/build_dictionary.py $data_dir/$prefix.$tgt.tok.bpe

# dev : bpe
python $bpe_scripts/apply_bpe.py -c $data_dir/$prefix.$tgt.tok.codes --vocabulary $data_dir/$prefix.$tgt.tok.vocab \
    --vocabulary-threshold 20 < $data_dir/valid.$tgt.tok > $data_dir/valid.$tgt.tok.bpe
