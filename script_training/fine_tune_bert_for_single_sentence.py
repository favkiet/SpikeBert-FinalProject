import torch
import torch.nn as nn
import pickle
import argparse
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import time
from dataset import TxtDataset, CSVDataset
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
from torchmetrics.classification import MatthewsCorrCoef

def to_device(x, device):
    for key in x:
        x[key] = x[key].to(device)

def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",default="cola",type=str)
    parser.add_argument("--batch_size",default=50,type=int)
    parser.add_argument("--fine_tune_lr",default=5e-5,type=float)
    parser.add_argument("--epochs",default=4,type=int)
    parser.add_argument("--teacher_model_name",default="bert-base-cased",type=str)
    parser.add_argument("--model_type",default="bert",type=str,choices=["bert", "modernbert"])
    parser.add_argument("--label_num",default=2,type=int)
    parser.add_argument("--metric", default="acc", type=str)
    parser.add_argument("--max_length",default=512,type=int)
    parser.add_argument("--output_dir",default="saved_models",type=str)
    parser.add_argument("--use_csv",default=False,type=bool)
    parser.add_argument("--sentence_col",default="sentence",type=str)
    parser.add_argument("--label_col",default="sentiment",type=str)
    args = parser.parse_args()
    return args

def get_model_and_tokenizer(model_name, model_type, num_labels):
    """Lấy model và tokenizer dựa trên loại model"""
    if model_type == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    elif model_type == "modernbert":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return tokenizer, model

def fine_tune_teacher_model(args):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Lấy model và tokenizer
    tokenizer, model = get_model_and_tokenizer(args.teacher_model_name, args.model_type, args.label_num)
    optimizer = AdamW(model.parameters(), lr=args.fine_tune_lr)
    
    # Chọn dataset class dựa trên tham số use_csv
    if args.use_csv:
        train_dataset = CSVDataset(data_path=f"/kaggle/input/{args.dataset_name}/train.csv", 
                                  sentence_col=args.sentence_col, 
                                  label_col=args.label_col)
        test_dataset = CSVDataset(data_path=f"/kaggle/input/{args.dataset_name}/test.csv",
                                 sentence_col=args.sentence_col, 
                                 label_col=args.label_col)
        # Kiểm tra xem có file validation không
        validation_path = f"/kaggle/input/{args.dataset_name}/validation.csv"
        if os.path.exists(validation_path):
            valid_dataset = CSVDataset(data_path=validation_path,
                                     sentence_col=args.sentence_col, 
                                     label_col=args.label_col)
        else:
            valid_dataset = test_dataset  # Sử dụng test làm validation nếu không có
    else:
        train_dataset = TxtDataset(data_path=f"/kaggle/input/{args.dataset_name}/train.txt")
        test_dataset = TxtDataset(data_path=f"/kaggle/input/{args.dataset_name}/test.txt")
        # Kiểm tra xem có file validation không
        validation_path = f"/kaggle/input/{args.dataset_name}/validation.txt"
        if os.path.exists(validation_path):
            valid_dataset = TxtDataset(data_path=validation_path)
        else:
            valid_dataset = test_dataset  # Sử dụng test làm validation nếu không có
    
    train_data_loader = DataLoader(dataset=train_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=True, 
                                  drop_last=False)
    test_data_loader = DataLoader(dataset=test_dataset, 
                                 batch_size=args.batch_size, 
                                 shuffle=False,
                                 drop_last=False)
    valid_data_loader = DataLoader(dataset=valid_dataset, 
                                  batch_size=args.batch_size, 
                                  shuffle=False,
                                  drop_last=False)
    
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(f"Available devices: {device_ids}")
    print(f"Using model: {args.teacher_model_name} (type: {args.model_type})")
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids).to(device)
    model = model.to(device=device)
    model.train()
    
    # Tạo thư mục output nếu chưa có
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in tqdm(range(args.epochs)):
        loss_list = []
        for i, batch in enumerate(train_data_loader):
            inputs = tokenizer(batch[0], padding=True, truncation=True, return_tensors="pt", max_length=args.max_length)
            labels = batch[1].to(device)
            to_device(inputs, device)
            outputs = model(**inputs)
            loss = F.cross_entropy(outputs.logits, labels)
            loss_list.append(loss.item())
            
            if i % 100 == 0:  # In loss mỗi 100 batch
                print(f"Epoch {epoch}, Batch {i}, Loss: {torch.mean(torch.tensor(loss_list)).item():.4f}")
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluation
        y_true = []
        y_pred = []
        with torch.no_grad():
            model.eval()
            for batch in valid_data_loader:
                b_y = batch[1]
                y_true.extend(b_y.to("cpu").tolist())
                input_dict = tokenizer(batch[0], return_tensors='pt', padding=True, truncation=True, max_length=args.max_length)
                to_device(input_dict, device)
                output = (model(**input_dict).logits).to("cpu")
                y_pred.extend(torch.max(output,1)[1].tolist())
        
        if args.metric == "acc":
            correct = 0
            for i in range(len(y_true)):
                correct += 1 if y_true[i] == y_pred[i] else 0
            acc = correct / len(y_pred)
            print(f"Epoch {epoch}, Validation Accuracy: {acc:.4f}")
        elif args.metric == "mcc":
            matthews_corrcoef = MatthewsCorrCoef(task='binary')
            mcc = matthews_corrcoef(torch.tensor(y_true), torch.tensor(y_pred))
            print(f"Epoch {epoch}, Validation MCC: {mcc:.4f}")
        
        # Lưu model
        current_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        record = acc if args.metric == "acc" else mcc
        model_save_path = f"{args.output_dir}/{args.model_type}_{args.dataset_name}_epoch{epoch}_{record:.4f}_{current_time}"
        
        tokenizer.save_pretrained(model_save_path)
        if len(device_ids) <= 1:
            model.save_pretrained(model_save_path)
        else:
            model.module.save_pretrained(model_save_path)
        
        print(f"Model saved to: {model_save_path}")
    
    # Lưu model cuối cùng
    final_model_path = f"{args.output_dir}/{args.model_type}_{args.dataset_name}_final"
    tokenizer.save_pretrained(final_model_path)
    if len(device_ids) <= 1:
        model.save_pretrained(final_model_path)
    else:
        model.module.save_pretrained(final_model_path)
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _args = args()
    fine_tune_teacher_model(_args)