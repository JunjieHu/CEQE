#!/bin/sh

export CUDA_VISIBLE_DEVICES=$1

langs="en_lv"
mt="nmt"
trn_dev_path="data/WMT2018QE/word-level/$langs.${mt}/"
tst_path=$trn_dev_path
baseline_feature_path="data/WMT2018QE/features/$langs/word_features/"
suffix="src\ mt\ src-mt.alignments\ tags\ pe\ ref\ src_tags\ ${mt}.features"

# Create vocabulary
vocab="$trn_dev_path/vocab.pos.${mt}.bin"
if [ ! -f $vocab ]; then
echo "create new vocab file"
python src/vocab.py \
    --train_src "$trn_dev_path/train.src" \
    --train_trg "$trn_dev_path/train.mt" \
    --train_feature "$baseline_feature_path/train.${mt}.features" \
    --include_singleton \
    --share_vocab \
    --output $vocab
fi

model="CrosslingualConv"
for embed_size in 64; do 
    for valid_niter in 100; do

output_dir="output/WMT2018QE/word-level/$langs$mt/submission-v1/$model-embed${embed_size}-valid${valid_niter}"
mkdir -p $output_dir

model_file="$output_dir/model"
save_submission="$output_dir/submission"
log_file="$output_dir/train.log"



# Training
CMD="python src/main.py \
    --mode train \
    --model $model \
    --vocab $vocab \
    --save_model $model_file \
    --valid_niter $valid_niter \
    --valid_metric f1-multi \
    --uniform_init 0.1 \
    --trn_dev_path $trn_dev_path \
    --tst_path $tst_path \
    --suffix $suffix \
    --dropout 0.3 \
    --clip_grad 5.0 \
    --lr_decay 0.75 \
    --lr 0.001 \
    --batch_size 64\
    --cuda\
    --baseline_feature_path $baseline_feature_path \
    --extra_feat_size 0 \
    --embed_size ${embed_size} \
    --hidden_size ${embed_size}  \
    --save_submission $save_submission \
    --submission_name Conv${embed_size} \
    --conv_size 64 \
    --skip_gap \
    --share_vocab"

echo "$CMD > ${log_file}"
echo "$CMD" > ${log_file}
bash -c "$CMD" >> ${log_file}
done 
done
