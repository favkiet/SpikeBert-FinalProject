import os
import pandas as pd
import glob

def convert_txt_to_csv(input_dir, output_dir):
    """Chuyển đổi các file txt sang csv"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Tìm tất cả file txt trong thư mục input
    txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
    
    for txt_file in txt_files:
        filename = os.path.basename(txt_file)
        base_name = os.path.splitext(filename)[0]
        
        print(f"Đang chuyển đổi {filename}...")
        
        # Đọc file txt
        sentences = []
        labels = []
        
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Bỏ qua dòng trống
                    # Tách sentence và label bằng tab
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        sentence = parts[0]
                        label = parts[1]
                        sentences.append(sentence)
                        labels.append(label)
        
        # Tạo DataFrame
        df = pd.DataFrame({
            'sentence': sentences,
            'sentiment': labels
        })
        
        # Lưu file CSV
        csv_file = os.path.join(output_dir, f"{base_name}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"Đã lưu {csv_file} với {len(df)} mẫu")
        print(f"Phân bố sentiment: {df['sentiment'].value_counts().to_dict()}")
        print("-" * 50)

def main():
    """Hàm chính"""
    # Chuyển đổi UIT-VSFC
    print("Chuyển đổi dữ liệu UIT-VSFC...")
    convert_txt_to_csv("data/UIT-VSFC", "data/UIT-VSFC")
    
    # Chuyển đổi UIT
    print("\nChuyển đổi dữ liệu UIT...")
    convert_txt_to_csv("data/uit", "data/uit")
    
    # Chuyển đổi SST (nếu có)
    if os.path.exists("data/sst"):
        print("\nChuyển đổi dữ liệu SST...")
        convert_txt_to_csv("data/sst", "data/sst")
    
    print("\nHoàn thành chuyển đổi!")

if __name__ == "__main__":
    main() 