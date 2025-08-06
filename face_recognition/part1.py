# Question
# Thực hiện các bước tiền xử lý dữ liệu ảnh để chuẩn bị cho mô hình SVM (Support Vector Machine).
# Bao gồm: tải ảnh, trực quan hóa, chuẩn hóa dữ liệu (mean-variance normalization), và định dạng dữ liệu phù hợp cho SVM.
# Hiểu rõ cấu trúc của các biến Xtrain, ytrain, Xval, yval để sử dụng trong các bước tiếp theo.
#Xtrain_flat, ytrain: Dữ liệu huấn luyện đã chuẩn hóa và flatten.
#Xval_flat, yval: Dữ liệu validation đã chuẩn hóa và flatten.

import matplotlib.pyplot as plt
#Step1:

plt.figure(figsize=(10, 4))
for i in range(5):  # Hiển thị 5 ảnh đầu tiên
    plt.subplot(1, 5, i+1)
    plt.imshow(Xtrain[i])
    plt.title(f'Label: {ytrain[i]}')
    plt.axis('off')
plt.show()

#Step2:
# Tính mean và std của tập huấn luyện
mean = np.mean(Xtrain, axis=(0, 1, 2))  # Mean theo từng kênh màu
std = np.std(Xtrain, axis=(0, 1, 2))    # Std theo từng kênh màu

# Chuẩn hóa tập huấn luyện và validation
Xtrain_norm = (Xtrain - mean) / std
Xval_norm = (Xval - mean) / std

#Step3:
# Reshape ảnh thành vector
n_train = Xtrain_norm.shape[0]
n_val = Xval_norm.shape[0]

# Flatten ảnh: (n_samples, height, width, channels) -> (n_samples, height*width*channels)
Xtrain_flat = Xtrain_norm.reshape(n_train, -1)
Xval_flat = Xval_norm.reshape(n_val, -1)