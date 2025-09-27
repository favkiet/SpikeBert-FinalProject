python -u train_spikformer.py \
        --seed 42 \
        --dataset_name uit \
        --label_num 3 \
        --batch_size 32 \
        --fine_tune_lr 1e-4 \
        --epochs 100 \
        --depths 6 \
        --max_length 128 \
        --dim 768 \
        --tau 10.0 \
        --common_thr 1.0 \
        --num_step 32 \
        --tokenizer_path "bert-base-multilingual-cased" \
        > "train.log"