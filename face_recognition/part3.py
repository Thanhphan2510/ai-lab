# Quesion
# Triển khai phát hiện khuôn mặt bằng SVM tuyến tính kết hợp sliding window và Non-Maxima Suppression (NMS).
# Thử nghiệm với các ngưỡng confidence khác nhau để tối ưu hóa kết quả.
# Đánh giá hiệu năng trên 4 ảnh test (img1.jpg - img4.jpg).

# Main step
# Sliding Window: Quét ảnh bằng các cửa sổ trượt chồng lấn.
# Phân loại SVM: Dùng SVM tuyến tính để tính confidence score cho từng patch.
# NMS: Loại bỏ các detection trùng lặp quanh cùng 1 khuôn mặt.
# Tối ưu ngưỡng: Thử nghiệm các giá trị confthresh (ngưỡng tiền lọc) và confthreshnms (ngưỡng hậu lọc).

#Step1
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

# Load mô hình SVM đã huấn luyện
svm = LinearSVC()
svm.fit(Xtrain, ytrain)  # Giả sử đã có dữ liệu train

# Tham số sliding window
window_size = (64, 64)  # Kích thước patch (giống tập train)
step = 10                # Bước trượt (pixel)
scales = [0.8, 1.0, 1.2] # Tỷ lệ đa tỷ lệ (multi-scale)

# Step2:
# Non-Maxima Suppression (NMS) function
def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return 

# Step3:
def detect_faces(img, svm, confthresh=1.0, confthreshnms=1.5):
    orig_img = img.copy()
    all_boxes = []
    all_confidences = []
    
    # Xử lý đa tỷ lệ (multi-scale)
    for scale in scales:
        scaled_img = cv2.resize(img, None, fx=scale, fy=scale)
        if scaled_img.shape[0] < window_size[0] or scaled_img.shape[1] < window_size[1]:
            continue
            
        # Trích xuất patches bằng sliding window
        for y in range(0, scaled_img.shape[0] - window_size[0], step):
            for x in range(0, scaled_img.shape[1] - window_size[1], step):
                patch = scaled_img[y:y+window_size[0], x:x+window_size[1]]
                patch = cv2.resize(patch, (64, 64))  # Chuẩn hóa kích thước
                
                # Tiền xử lý và dự đoán
                patch_norm = (patch - mean) / std  # Chuẩn hóa
                patch_flat = patch_norm.reshape(1, -1)
                confidence = svm.decision_function(patch_flat)[0]
                
                # Lưu kết quả đạt ngưỡng confthresh
                if confidence >= confthresh:
                    orig_x = int(x / scale)
                    orig_y = int(y / scale)
                    orig_w = int(window_size[1] / scale)
                    orig_h = int(window_size[0] / scale)
                    all_boxes.append([orig_x, orig_y, orig_w, orig_h])
                    all_confidences.append(confidence)
    
    # Áp dụng NMS
    if len(all_boxes) > 0:
        boxes = np.array(all_boxes)
        confidences = np.array(all_confidences)
        keep = nms(boxes, confidences)
        
        # Lọc kết quả sau NMS bằng confthreshnms
        final_boxes = []
        for i in keep:
            if confidences[i] >= confthreshnms:
                final_boxes.append(boxes[i])
        return final_boxes
    return []

# Step4: Test thresholds
# Thử nghiệm các ngưỡng
confthresh_options = [0.5, 1.0, 1.5]
confthreshnms_options = [1.0, 1.5, 2.0]
images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']

for img_path in images:
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    
    for i, confthresh in enumerate(confthresh_options):
        for j, confthreshnms in enumerate(confthreshnms_options):
            boxes = detect_faces(img, svm, confthresh, confthreshnms)
            
            # Vẽ bounding boxes
            result_img = img.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Hiển thị kết quả
            plt.subplot(len(confthresh_options), len(confthreshnms_options), i*len(confthreshnms_options)+j+1)
            plt.imshow(result_img)
            plt.title(f'c1={confthresh}, c2={confthreshnms}')
            plt.axis('off')
    
    plt.suptitle(f'Kết quả cho {img_path}')
    plt.show()

# Step5: Analysis results
# Hiệu ứng của confthresh (ngưỡng tiền lọc):
# Giá trị thấp (0.5): Phát hiện nhiều khuôn mặt nhưng nhiều false positives
# Giá trị cao (1.5): Ít false positives nhưng có thể bỏ sót khuôn mặt thật

# Hiệu ứng của confthreshnms (ngưỡng hậu lọc):
# Giá trị thấp (1.0): Giữ lại nhiều detection sau NMS
# Giá trị cao (2.0): Chỉ giữ lại detection có độ tin cậy rất cao

# Câu hỏi: "Có ngưỡng NMS duy nhất hoạt động hoàn hảo cho mọi ảnh?"
# Không thể, vì:
# Ảnh khác nhau có điều kiện ánh sáng/độ phân giải khác nhau
# Kích thước và hướng mặt khác nhau
# Độ phức tạp nền khác nhau (ví dụ: img4.jpg có nhiều vật thể gây nhiễu)

# Giải pháp thực tế:
# Huấn luyện mô hình SVM trên dataset đa dạng
# Sử dụng HOG (Histogram of Oriented Gradients) thay vì raw pixels
# Kết hợp nhiều kỹ thuật (CNN, Data Augmentation)

# Step6: Explain NMS role
# Minh họa NMS (trước và sau)
img = cv2.imread('img2.jpg')
boxes_pre_nms = detect_faces(img, svm, confthresh=0.5, confthreshnms=-10)  # Tắt NMS
boxes_post_nms = detect_faces(img, svm, confthresh=0.5, confthreshnms=1.0)

# Vẽ kết quả
plt.figure(figsize=(10, 5))
plt.subplot(121)
for (x, y, w, h) in boxes_pre_nms:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
plt.title(f'Trước NMS: {len(boxes_pre_nms)} boxes')
plt.axis('off')

plt.subplot(122)
for (x, y, w, h) in boxes_post_nms:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
plt.title(f'Sau NMS: {len(boxes_post_nms)} boxes')
plt.axis('off')