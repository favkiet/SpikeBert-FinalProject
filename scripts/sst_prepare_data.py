import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset

def convert_label(label):
    """Chuyển đổi label từ float sang sentiment"""
    if label <= 0.4:
        return "negative"
    elif label >= 0.6:
        return "positive"
    else:
        return "neutral"

def save_sst_to_files(data_dict, output_dir):
    """Lưu dữ liệu SST vào các file txt và CSV theo format SpikeBERT"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Chuẩn bị dữ liệu
    sentences = []
    labels = []
    
    for item in data_dict:
        sentence = item['sentence']
        label = convert_label(item['label'])
        sentences.append(sentence)
        labels.append(label)
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'sentence': sentences,
        'sentiment': labels
    })
    
    # Chia dữ liệu thành train (80%), validation (10%), test (10%)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['sentiment'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['sentiment'])
    
    
    # Hàm lưu dữ liệu vào file CSV
    def save_to_csv(dataframe, filename):
        dataframe.to_csv(os.path.join(output_dir, filename), index=False, encoding='utf-8')
    # Lưu dữ liệu vào các file CSV
    save_to_csv(train_df, 'train.csv')
    save_to_csv(val_df, 'validation.csv')
    save_to_csv(test_df, 'test.csv')
    
    # Lưu file CSV tổng hợp
    df.to_csv(os.path.join(output_dir, 'sst_dataset.csv'), index=False, encoding='utf-8')
    
    print(f"Đã lưu dữ liệu SST vào: {output_dir}")
    print(f"Tổng số mẫu: {len(df)}")
    print(f"Số mẫu train: {len(train_df)}")
    print(f"Số mẫu validation: {len(val_df)}")
    print(f"Số mẫu test: {len(test_df)}")
    print(f"Phân bố sentiment:")
    print(df['sentiment'].value_counts())
    
    # In ra danh sách file đã tạo
    print(f"\nCác file đã tạo:")
    print(f"- {output_dir}/train.txt")
    print(f"- {output_dir}/validation.txt") 
    print(f"- {output_dir}/test.txt")
    print(f"- {output_dir}/train.csv")
    print(f"- {output_dir}/validation.csv")
    print(f"- {output_dir}/test.csv")
    print(f"- {output_dir}/sst_dataset.csv")

def main():
    """Hàm chính để chuẩn bị dữ liệu SST"""
    print("Bắt đầu chuẩn bị dữ liệu SST...")
    
    # Tạo thư mục data/sst nếu chưa tồn tại
    os.makedirs("data/sst", exist_ok=True)
    
    try:
        # Load dataset SST
        print("Đang tải dataset SST...")
        dataset = load_dataset("sst", "default")
        
        # Thu thập tất cả dữ liệu
        all_data = []
        
        print("Đang thu thập dữ liệu từ train split...")
        for item in dataset['train']:
            all_data.append(item)
        
        print("Đang thu thập dữ liệu từ validation split...")
        for item in dataset['validation']:
            all_data.append(item)
        
        print("Đang thu thập dữ liệu từ test split...")
        for item in dataset['test']:
            all_data.append(item)
        
        print(f"Tổng cộng thu thập được {len(all_data)} mẫu")
        
        # Lưu dữ liệu
        save_sst_to_files(all_data, "data/sst")
        
        print("Hoàn thành chuẩn bị dữ liệu SST!")
        
    except Exception as e:
        print(f"Lỗi khi chuẩn bị dữ liệu: {e}")
        print("Đảm bảo đã cài đặt datasets library: pip install datasets")

if __name__ == "__main__":
    main() 