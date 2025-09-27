import os
import pandas as pd
from sklearn.model_selection import train_test_split

def convert_label(label):
    """Chuyển đổi label từ int sang sentiment cho Yelp"""
    if label == 1:
        return "negative"
    elif label == 2:
        return "positive"
    else:
        return "unknown"

def save_yelp_to_files(train_df, test_df, output_dir):
    """Lưu dữ liệu Yelp vào các file txt và CSV theo format SpikeBERT"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Hàm lưu dữ liệu vào file CSV
    def save_to_csv(dataframe, filename):
        dataframe.to_csv(os.path.join(output_dir, filename), index=False, encoding='utf-8')

    
    # Lưu dữ liệu vào các file CSV
    save_to_csv(train_df, 'train.csv')
    save_to_csv(test_df, 'test.csv')

    
    print(f"Đã lưu dữ liệu Yelp vào: {output_dir}")
    print(f"Số mẫu test: {len(test_df)}")
    print(f"Phân bố sentiment:")

    # In ra danh sách file đã tạo
    print(f"\nCác file đã tạo:")
    print(f"- {output_dir}/train.csv")
    print(f"- {output_dir}/test.csv")

def main():
    """Hàm chính để chuẩn bị dữ liệu Yelp"""
    print("Bắt đầu chuẩn bị dữ liệu Yelp Review Polarity...")
    
    # Đường dẫn file
    train_file = "data/yelp_review_polarity_csv/train.csv"
    test_file = "data/yelp_review_polarity_csv/test.csv"
    
    # Tạo thư mục data/yelp nếu chưa tồn tại
    os.makedirs("data/yelp", exist_ok=True)
    
    try:
        # Đọc file train.csv
        print("Đang đọc file train.csv...")
        train_df = pd.read_csv(train_file, header=None, names=['label', 'sentence'])
        
        # Đọc file test.csv
        print("Đang đọc file test.csv...")
        test_df = pd.read_csv(test_file, header=None, names=['label', 'sentence'])
        
        print(f"Train data: {len(train_df)} mẫu")
        print(f"Test data: {len(test_df)} mẫu")
        
        # Chuyển đổi label
        print("Đang chuyển đổi label...")
        train_df['sentiment'] = train_df['label'].apply(convert_label)
        test_df['sentiment'] = test_df['label'].apply(convert_label)
        
        # Lưu dữ liệu
        save_yelp_to_files(train_df, test_df, "data/yelp")
        
        print("Hoàn thành chuẩn bị dữ liệu Yelp!")
        
        # Thống kê thêm
        print(f"\nThống kê chi tiết:")
        print(f"Train - Negative: {len(train_df[train_df['sentiment'] == 'negative'])}")
        print(f"Train - Positive: {len(train_df[train_df['sentiment'] == 'positive'])}")
        print(f"Test - Negative: {len(test_df[test_df['sentiment'] == 'negative'])}")
        print(f"Test - Positive: {len(test_df[test_df['sentiment'] == 'positive'])}")
        
    except Exception as e:
        print(f"Lỗi khi chuẩn bị dữ liệu: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 