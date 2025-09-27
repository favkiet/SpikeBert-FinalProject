# SpikeBERT Training Guide

Hướng dẫn training SpikeBERT cho các dataset sentiment analysis khác nhau.

## Cài đặt môi trường

```bash
# Cài đặt các thư viện cần thiết
pip install torch transformers datasets pandas scikit-learn tqdm
pip install spikingjelly
```

## Cấu trúc dữ liệu

Các dataset được hỗ trợ:
- **UIT-VSFC**: Dữ liệu feedback sinh viên UIT (tiếng Việt)
- **UIT**: Dữ liệu feedback sinh viên UIT (tiếng Việt) 
- **SST**: Stanford Sentiment Treebank (3 nhãn: negative, neutral, positive)
- **SST-2**: Stanford Sentiment Treebank v2 (2 nhãn: negative, positive)
- **Yelp**: Yelp Review Polarity (2 nhãn: negative, positive)

## Models được hỗ trợ

### 1. BERT Models
- **bert-base-cased**: BERT cơ bản cho tiếng Anh
- **bert-base-multilingual-cased**: BERT đa ngôn ngữ (hỗ trợ tiếng Việt)
- **answerdotai/ModernBERT-base**: ModernBERT - phiên bản cải tiến của BERT

### 2. Model Types
- **bert**: Sử dụng BertTokenizer và BertForSequenceClassification
- **modernbert**: Sử dụng AutoTokenizer và AutoModelForSequenceClassification

## Chuẩn bị dữ liệu

### 1. UIT-VSFC Dataset

```bash
cd scripts
python uit_vsfc_prepare_data.py
```

**Kết quả**: Tạo thư mục `data/UIT-VSFC/` với các file:
- `train.csv`, `test.csv` - Dữ liệu với cột `sentence`, `sentiment`
- `train.txt`, `test.txt` - Format cho SpikeBERT (sentence\tlabel)

### 2. UIT Dataset

```bash
cd scripts
python prepare_data.py
```

**Kết quả**: Tạo thư mục `data/uit/` với các file:
- `train.csv`, `validation.csv`, `test.csv`
- `train.txt`, `validation.txt`, `test.txt`

### 3. SST Dataset

```bash
cd scripts
python sst_prepare_data.py
```

**Kết quả**: Tạo thư mục `data/sst/` với các file:
- `train.csv`, `validation.csv`, `test.csv`
- `train.txt`, `validation.txt`, `test.txt`
- 3 nhãn: negative, neutral, positive

### 4. SST-2 Dataset

```bash
cd scripts
python sst2_prepare_data.py
```

**Kết quả**: Tạo thư mục `data/sst2/` với các file:
- `train.csv`, `validation.csv`, `test.csv`
- `train.txt`, `validation.txt`, `test.txt`
- 2 nhãn: negative, positive

### 5. Yelp Dataset

```bash
cd scripts
python yelp_prepare_data.py
```

**Kết quả**: Tạo thư mục `data/yelp/` với các file:
- `train.csv`, `test.csv`
- `train.txt`, `test.txt`
- 2 nhãn: negative, positive

## Training SpikeBERT

### Workflow chung

1. **Fine-tune BERT** (teacher model)
2. **Pre-training SpikeBERT** (nếu cần)
3. **Knowledge Distillation** (chuyển kiến thức từ BERT sang SpikeBERT)
4. **Training trực tiếp** (so sánh hiệu quả)

### Dataset Classes

Hệ thống hỗ trợ 2 loại dataset:
- **TxtDataset**: Đọc từ file `.txt` (format: sentence\tlabel)
- **CSVDataset**: Đọc từ file `.csv` (format: pandas DataFrame)

### 1. UIT-VSFC Dataset

```bash
cd SpikeBERT

# Fine-tune BERT (Traditional)
python -u fine_tune_bert_for_single_sentence.py \
    --dataset_name UIT-VSFC \
    --label_num 3 \
    --batch_size 32 \
    --fine_tune_lr 2e-5 \
    --epochs 10 \
    --max_length 128 \
    --teacher_model_name "bert-base-multilingual-cased" \
    --model_type "bert" \
    --output_dir "saved_models/bert_uit_vsfc" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/finetune_uit_vsfc.log"

# Fine-tune ModernBERT
python -u fine_tune_bert_for_single_sentence.py \
    --dataset_name UIT-VSFC \
    --label_num 3 \
    --batch_size 32 \
    --fine_tune_lr 2e-5 \
    --epochs 10 \
    --max_length 128 \
    --teacher_model_name "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --output_dir "saved_models/modernbert_uit_vsfc" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/finetune_modernbert_uit_vsfc.log"

# Knowledge Distillation (BERT)
python -u new_distill_spikformer.py \
    --dataset_name UIT-VSFC \
    --label_num 3 \
    --batch_size 32 \
    --fine_tune_lr 1e-4 \
    --epochs 50 \
    --depths 6 \
    --max_length 128 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --teacher_model_path "bert-base-multilingual-cased" \
    --model_type "bert" \
    --predistill_model_path "saved_models/bert_uit_vsfc" \
    --temperature 2.0 \
    --alpha 0.5 \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/distill_uit_vsfc.log"

# Knowledge Distillation (ModernBERT)
python -u new_distill_spikformer.py \
    --dataset_name UIT-VSFC \
    --label_num 3 \
    --batch_size 32 \
    --fine_tune_lr 1e-4 \
    --epochs 50 \
    --depths 6 \
    --max_length 128 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --teacher_model_path "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --predistill_model_path "saved_models/modernbert_uit_vsfc" \
    --temperature 2.0 \
    --alpha 0.5 \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/distill_modernbert_uit_vsfc.log"

# Training trực tiếp (BERT)
python -u train_spikformer.py \
    --dataset_name UIT-VSFC \
    --label_num 3 \
    --batch_size 16 \
    --fine_tune_lr 1e-4 \
    --epochs 100 \
    --depths 6 \
    --max_length 128 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --tokenizer_path "bert-base-multilingual-cased" \
    --model_type "bert" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/train_spikebert_uit_vsfc.log"

# Training trực tiếp (ModernBERT)
python -u train_spikformer.py \
    --dataset_name UIT-VSFC \
    --label_num 3 \
    --batch_size 16 \
    --fine_tune_lr 1e-4 \
    --epochs 100 \
    --depths 6 \
    --max_length 128 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --tokenizer_path "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/train_spikebert_modernbert_uit_vsfc.log"
```

### 2. SST Dataset

```bash
cd SpikeBERT

# Fine-tune BERT
python -u fine_tune_bert_for_single_sentence.py \
    --dataset_name sst \
    --label_num 3 \
    --batch_size 32 \
    --fine_tune_lr 2e-5 \
    --epochs 10 \
    --max_length 64 \
    --teacher_model_name "bert-base-cased" \
    --model_type "bert" \
    --output_dir "saved_models/bert_sst" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/finetune_sst.log"

# Fine-tune ModernBERT
python -u fine_tune_bert_for_single_sentence.py \
    --dataset_name sst \
    --label_num 3 \
    --batch_size 32 \
    --fine_tune_lr 2e-5 \
    --epochs 10 \
    --max_length 64 \
    --teacher_model_name "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --output_dir "saved_models/modernbert_sst" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/finetune_modernbert_sst.log"

# Knowledge Distillation (BERT)
python -u new_distill_spikformer.py \
    --dataset_name sst \
    --label_num 3 \
    --batch_size 32 \
    --fine_tune_lr 1e-4 \
    --epochs 50 \
    --depths 6 \
    --max_length 64 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --teacher_model_path "bert-base-cased" \
    --model_type "bert" \
    --predistill_model_path "saved_models/bert_sst" \
    --temperature 2.0 \
    --alpha 0.5 \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/distill_sst.log"

# Knowledge Distillation (ModernBERT)
python -u new_distill_spikformer.py \
    --dataset_name sst \
    --label_num 3 \
    --batch_size 32 \
    --fine_tune_lr 1e-4 \
    --epochs 50 \
    --depths 6 \
    --max_length 64 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --teacher_model_path "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --predistill_model_path "saved_models/modernbert_sst" \
    --temperature 2.0 \
    --alpha 0.5 \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/distill_modernbert_sst.log"

# Training trực tiếp (BERT)
python -u train_spikformer.py \
    --dataset_name sst \
    --label_num 3 \
    --batch_size 16 \
    --fine_tune_lr 1e-4 \
    --epochs 100 \
    --depths 6 \
    --max_length 64 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --tokenizer_path "bert-base-cased" \
    --model_type "bert" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/train_spikebert_sst.log"

# Training trực tiếp (ModernBERT)
python -u train_spikformer.py \
    --dataset_name sst \
    --label_num 3 \
    --batch_size 16 \
    --fine_tune_lr 1e-4 \
    --epochs 100 \
    --depths 6 \
    --max_length 64 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --tokenizer_path "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/train_spikebert_modernbert_sst.log"
```

### 3. SST-2 Dataset

```bash
cd SpikeBERT

# Fine-tune BERT
python -u fine_tune_bert_for_single_sentence.py \
    --dataset_name sst2 \
    --label_num 2 \
    --batch_size 32 \
    --fine_tune_lr 2e-5 \
    --epochs 10 \
    --max_length 64 \
    --teacher_model_name "bert-base-cased" \
    --model_type "bert" \
    --output_dir "saved_models/bert_sst2" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/finetune_sst2.log"

# Fine-tune ModernBERT
python -u fine_tune_bert_for_single_sentence.py \
    --dataset_name sst2 \
    --label_num 2 \
    --batch_size 32 \
    --fine_tune_lr 2e-5 \
    --epochs 10 \
    --max_length 64 \
    --teacher_model_name "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --output_dir "saved_models/modernbert_sst2" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/finetune_modernbert_sst2.log"

# Knowledge Distillation (BERT)
python -u new_distill_spikformer.py \
    --dataset_name sst2 \
    --label_num 2 \
    --batch_size 32 \
    --fine_tune_lr 1e-4 \
    --epochs 50 \
    --depths 6 \
    --max_length 64 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --teacher_model_path "bert-base-cased" \
    --model_type "bert" \
    --predistill_model_path "saved_models/bert_sst2" \
    --temperature 2.0 \
    --alpha 0.5 \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/distill_sst2.log"

# Knowledge Distillation (ModernBERT)
python -u new_distill_spikformer.py \
    --dataset_name sst2 \
    --label_num 2 \
    --batch_size 32 \
    --fine_tune_lr 1e-4 \
    --epochs 50 \
    --depths 6 \
    --max_length 64 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --teacher_model_path "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --predistill_model_path "saved_models/modernbert_sst2" \
    --temperature 2.0 \
    --alpha 0.5 \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/distill_modernbert_sst2.log"

# Training trực tiếp (BERT)
python -u train_spikformer.py \
    --dataset_name sst2 \
    --label_num 2 \
    --batch_size 16 \
    --fine_tune_lr 1e-4 \
    --epochs 100 \
    --depths 6 \
    --max_length 64 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --tokenizer_path "bert-base-cased" \
    --model_type "bert" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/train_spikebert_sst2.log"

# Training trực tiếp (ModernBERT)
python -u train_spikformer.py \
    --dataset_name sst2 \
    --label_num 2 \
    --batch_size 16 \
    --fine_tune_lr 1e-4 \
    --epochs 100 \
    --depths 6 \
    --max_length 64 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --tokenizer_path "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/train_spikebert_modernbert_sst2.log"
```

### 4. Yelp Dataset

```bash
cd SpikeBERT

# Fine-tune BERT
python -u fine_tune_bert_for_single_sentence.py \
    --dataset_name yelp \
    --label_num 2 \
    --batch_size 32 \
    --fine_tune_lr 2e-5 \
    --epochs 10 \
    --max_length 128 \
    --teacher_model_name "bert-base-cased" \
    --model_type "bert" \
    --output_dir "saved_models/bert_yelp" \
    --use_csv True \
    --sentence_col "review" \
    --label_col "sentiment" \
    > "log/finetune_yelp.log"

# Fine-tune ModernBERT
python -u fine_tune_bert_for_single_sentence.py \
    --dataset_name yelp \
    --label_num 2 \
    --batch_size 32 \
    --fine_tune_lr 2e-5 \
    --epochs 10 \
    --max_length 128 \
    --teacher_model_name "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --output_dir "saved_models/modernbert_yelp" \
    --use_csv True \
    --sentence_col "review" \
    --label_col "sentiment" \
    > "log/finetune_modernbert_yelp.log"

# Knowledge Distillation (BERT)
python -u new_distill_spikformer.py \
    --dataset_name yelp \
    --label_num 2 \
    --batch_size 32 \
    --fine_tune_lr 1e-4 \
    --epochs 50 \
    --depths 6 \
    --max_length 128 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --teacher_model_path "bert-base-cased" \
    --model_type "bert" \
    --predistill_model_path "saved_models/bert_yelp" \
    --temperature 2.0 \
    --alpha 0.5 \
    --use_csv True \
    --sentence_col "review" \
    --label_col "sentiment" \
    > "log/distill_yelp.log"

# Knowledge Distillation (ModernBERT)
python -u new_distill_spikformer.py \
    --dataset_name yelp \
    --label_num 2 \
    --batch_size 32 \
    --fine_tune_lr 1e-4 \
    --epochs 50 \
    --depths 6 \
    --max_length 128 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --teacher_model_path "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --predistill_model_path "saved_models/modernbert_yelp" \
    --temperature 2.0 \
    --alpha 0.5 \
    --use_csv True \
    --sentence_col "review" \
    --label_col "sentiment" \
    > "log/distill_modernbert_yelp.log"

# Training trực tiếp (BERT)
python -u train_spikformer.py \
    --dataset_name yelp \
    --label_num 2 \
    --batch_size 16 \
    --fine_tune_lr 1e-4 \
    --epochs 100 \
    --depths 6 \
    --max_length 128 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --tokenizer_path "bert-base-cased" \
    --model_type "bert" \
    --use_csv True \
    --sentence_col "review" \
    --label_col "sentiment" \
    > "log/train_spikebert_yelp.log"

# Training trực tiếp (ModernBERT)
python -u train_spikformer.py \
    --dataset_name yelp \
    --label_num 2 \
    --batch_size 16 \
    --fine_tune_lr 1e-4 \
    --epochs 100 \
    --depths 6 \
    --max_length 128 \
    --dim 768 \
    --tau 10.0 \
    --common_thr 1.0 \
    --num_step 16 \
    --tokenizer_path "answerdotai/ModernBERT-base" \
    --model_type "modernbert" \
    --use_csv True \
    --sentence_col "sentence" \
    --label_col "sentiment" \
    > "log/train_spikebert_modernbert_yelp.log"
```

## Tham số quan trọng

### Model parameters:
- **model_type**: `bert` hoặc `modernbert`
- **teacher_model_name**: Tên model BERT (ví dụ: "bert-base-cased", "answerdotai/ModernBERT-base")
- **tokenizer_path**: Đường dẫn đến tokenizer (thường giống teacher_model_name)

### Dataset-specific parameters:
- **UIT-VSFC/UIT**: `label_num=3`, `teacher_model_name="bert-base-multilingual-cased"` hoặc `"answerdotai/ModernBERT-base"`
- **SST**: `label_num=3`, `teacher_model_name="bert-base-cased"` hoặc `"answerdotai/ModernBERT-base"`
- **SST-2**: `label_num=2`, `teacher_model_name="bert-base-cased"` hoặc `"answerdotai/ModernBERT-base"`
- **Yelp**: `label_num=2`, `teacher_model_name="bert-base-cased"` hoặc `"answerdotai/ModernBERT-base"`

### CSV Dataset parameters:
- **use_csv**: `True` để sử dụng CSVDataset, `False` để sử dụng TxtDataset
- **sentence_col**: Tên cột chứa text (mặc định: "sentence")
- **label_col**: Tên cột chứa label (mặc định: "sentiment")
- **Yelp**: `sentence_col="review"` (do cấu trúc dữ liệu khác)

### Performance parameters:
- **max_length**: 64 cho SST/SST-2, 128 cho UIT/Yelp
- **batch_size**: 16-32 tùy theo GPU memory
- **num_step**: 16 cho CPU, 32 cho GPU
- **use_cpu**: True nếu không có GPU

### Knowledge Distillation parameters:
- **temperature**: 2.0 (mặc định) - điều chỉnh độ mềm của softmax
- **alpha**: 0.5 (mặc định) - trọng số cho knowledge distillation loss

## So sánh Models

### BERT vs ModernBERT:
- **BERT**: Model truyền thống, ổn định, được test nhiều
- **ModernBERT**: Phiên bản cải tiến, có thể cho kết quả tốt hơn nhưng cần test

### Khuyến nghị:
1. **Tiếng Việt**: Sử dụng `bert-base-multilingual-cased` hoặc `answerdotai/ModernBERT-base`
2. **Tiếng Anh**: Sử dụng `bert-base-cased` hoặc `answerdotai/ModernBERT-base`
3. **Thử nghiệm**: So sánh kết quả giữa BERT và ModernBERT

## Theo dõi training

Kiểm tra các file log để theo dõi:
- Loss và accuracy trong quá trình training
- Thời gian training
- Lỗi nếu có

## Kết quả

Models được lưu trong thư mục `saved_models/` với tên:
- `bert_[dataset_name]/` - BERT đã fine-tune
- `modernbert_[dataset_name]/` - ModernBERT đã fine-tune
- `distilled_spikformer/` - SpikeBERT sau knowledge distillation
- `trained_spikformer/` - SpikeBERT training trực tiếp

## Lưu ý

1. Đảm bảo có đủ disk space cho dataset và models
2. Training trên GPU sẽ nhanh hơn nhiều so với CPU
3. Điều chỉnh batch_size nếu gặp lỗi memory
4. Có thể giảm epochs nếu muốn training nhanh hơn
5. CSVDataset tự động xử lý label mapping từ text sang số
6. Sử dụng `--use_csv True` để tận dụng các file CSV đã chuẩn bị
7. ModernBERT có thể cần điều chỉnh hyperparameters khác với BERT
8. So sánh kết quả giữa các model để chọn model tốt nhất 