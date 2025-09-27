python -u fine_tune_bert_for_single_sentence.py \
        --seed 42 \
        --dataset_name uit \
        --label_num 3 \
        --batch_size 32 \
        --fine_tune_lr 2e-5 \
    --epochs 10 \
        --max_length 128 \
        --bert_model "bert-base-multilingual-cased" \
        --output_dir "saved_models/bert_uit" \
        > "finetune.log"