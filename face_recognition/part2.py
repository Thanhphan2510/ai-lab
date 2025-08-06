#Question
# Huấn luyện SVM tuyến tính và tính siêu phẳng phân lớp W
# Giải thích hình dạng của W W khi biểu diễn dưới dạng ảnh.
# Tối ưu tham số C (regularization) dựa trên tập validation.
# Phân tích ảnh hưởng của C lên hình dạng của W.
# Thảo luận về việc sử dụng "average face" thay thế siêu phẳng W.

#ANSWER
# Siêu phẳng W trong SVM không phải là average face mà là bản đồ trọng số tối ưu cho phân lớp
# W giống khuôn mặt vì nó kế thừa đặc trưng từ các support vectors
# C nhỏ cho W "giống mặt" hơn do tính chất làm mịn của regularization
# Luôn chọn C tối ưu qua validation để cân bằng bias-variance

# Step1:
from sklearn.svm import LinearSVC
import numpy as np
import matplotlib.pyplot as plt

# Huấn luyện SVM
svm = LinearSVC(C=1.0, max_iter=10000, random_state=42)
svm.fit(Xtrain, ytrain)

# Lấy siêu phẳng W và bias b
W = svm.coef_.reshape((height, width, channels))  # Reshape về kích thước ảnh gốc
b = svm.intercept_[0]

# Tính confidence scores
train_confidence = svm.decision_function(Xtrain)
val_confidence = svm.decision_function(Xval)

# Tính độ chính xác
train_acc = svm.score(Xtrain, ytrain)
val_acc = svm.score(Xval, yval)

# Step2:
# Chuẩn hóa W để hiển thị
W_normalized = (W - W.min()) / (W.max() - W.min())

# Hiển thị ảnh W
plt.figure(figsize=(6,6))
plt.imshow(W_normalized)
plt.title("Siêu phẳng W")
plt.axis('off')
plt.show()

# Step3:

# Step4:
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
best_acc = 0
best_model = None

for C in C_values:
    # Huấn luyện SVM
    svm = LinearSVC(C=C, max_iter=10000, random_state=42)
    svm.fit(Xtrain, ytrain)
    
    # Tính độ chính xác validation
    val_acc = svm.score(Xval, yval)
    
    # Lưu mô hình tốt nhất
    if val_acc > best_acc:
        best_acc = val_acc
        best_model = svm
    
    # Minh họa W
    W = svm.coef_.reshape((height, width, channels))
    W_norm = (W - W.min()) / (W.max() - W.min())
    
    plt.figure()
    plt.imshow(W_norm)
    plt.title(f"C = {C}, Acc = {val_acc:.2f}")
    plt.axis('off')
    plt.show()