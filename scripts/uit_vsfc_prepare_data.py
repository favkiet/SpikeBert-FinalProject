import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Đọc dữ liệu từ CSV
df = pd.read_csv('data/uit_students_feedback.csv')

# Tạo thư mục data/uit nếu chưa tồn tại
dataset_dir = 'data/uit'
os.makedirs(dataset_dir, exist_ok=True)

# Mapping nhãn text sang số
label_mapping = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}

# Chuẩn bị dữ liệu
X = df['sentence'].values
y = df['sentiment_label'].map(label_mapping).values  # Chuyển nhãn text thành số

# Chia dữ liệu thành train (70%), validation (15%) và test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Hàm lưu dữ liệu vào file txt
def save_to_txt(sentences, labels, filename):
    with open(os.path.join(dataset_dir, filename), 'w', encoding='utf-8') as f:
        for sentence, label in zip(sentences, labels):
            # Format: sentence\tlabel
            f.write(f"{sentence}\t{label}\n")

# Lưu dữ liệu vào các file
save_to_txt(X_train, y_train, 'train.txt')
save_to_txt(X_val, y_val, 'validation.txt')
save_to_txt(X_test, y_test, 'test.txt')

print(f"Tổng số mẫu: {len(df)}")
print(f"Số mẫu train: {len(X_train)}")
print(f"Số mẫu validation: {len(X_val)}")
print(f"Số mẫu test: {len(X_test)}")

# In ra mapping để tham khảo sau này
print("\nLabel mapping:")
for text, num in label_mapping.items():
    print(f"{text} -> {num}") 