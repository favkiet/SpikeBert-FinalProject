import os
import pandas as pd
import glob
from tqdm import tqdm

def read_review_file(file_path):
    """Đọc nội dung file review"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        return content
    except Exception as e:
        print(f"Lỗi đọc file {file_path}: {e}")
        return None

def extract_rating_from_filename(filename):
    """Trích xuất rating từ tên file"""
    # Format: [id]_[rating].txt
    parts = filename.replace('.txt', '').split('_')
    if len(parts) >= 2:
        return int(parts[-1])  # Lấy phần cuối làm rating
    return None

def process_directory(base_dir, split_type):
    """Xử lý một thư mục (train hoặc test)"""
    data = []
    
    # Xử lý positive reviews
    pos_dir = os.path.join(base_dir, 'pos')
    if os.path.exists(pos_dir):
        pos_files = glob.glob(os.path.join(pos_dir, '*.txt'))
        print(f"Đang xử lý {len(pos_files)} positive reviews...")
        
        for file_path in tqdm(pos_files, desc=f"Processing {split_type} positive"):
            content = read_review_file(file_path)
            if content:
                filename = os.path.basename(file_path)
                rating = extract_rating_from_filename(filename)
                data.append({
                    'text': content,
                    'sentiment': 'positive',
                    'rating': rating,
                    'split': split_type,
                    'filename': filename
                })
    
    # Xử lý negative reviews
    neg_dir = os.path.join(base_dir, 'neg')
    if os.path.exists(neg_dir):
        neg_files = glob.glob(os.path.join(neg_dir, '*.txt'))
        print(f"Đang xử lý {len(neg_files)} negative reviews...")
        
        for file_path in tqdm(neg_files, desc=f"Processing {split_type} negative"):
            content = read_review_file(file_path)
            if content:
                filename = os.path.basename(file_path)
                rating = extract_rating_from_filename(filename)
                data.append({
                    'text': content,
                    'sentiment': 'negative',
                    'rating': rating,
                    'split': split_type,
                    'filename': filename
                })
    
    return data

def main():
    """Hàm chính để tổng hợp dữ liệu"""
    base_path = 'data/aclImdb'
    
    print("Bắt đầu tổng hợp dữ liệu IMDB...")
    
    # Xử lý train data
    train_data = process_directory(os.path.join(base_path, 'train'), 'train')
    
    # Xử lý test data
    test_data = process_directory(os.path.join(base_path, 'test'), 'test')

    
    # Tạo file train và test riêng biệt
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    train_df.to_csv('data/imdb_train.csv', index=False, encoding='utf-8')
    test_df.to_csv('data/imdb_test.csv', index=False, encoding='utf-8')
    
    print(f"Đã lưu train data vào: data/imdb_train.csv")
    print(f"Đã lưu test data vào: data/imdb_test.csv")
    
    # Hiển thị một số mẫu
    print(f"\nMẫu dữ liệu:")
    print(df.head())
    
    # Thống kê rating
    print(f"\nThống kê rating:")
    print(df['rating'].value_counts().sort_index())

if __name__ == "__main__":
    main() 