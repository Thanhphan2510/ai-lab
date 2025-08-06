import warnings
import numpy as np
import cv2
import os
import scipy.io
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import defaultdict
import time
from scipy.stats import gaussian_kde

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------
# HOG & Sliding Window Parameters
# -----------------------------------
HOG_PARAMS = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'transform_sqrt': True,
    'feature_vector': True,
}

# Absolute face size limits
ABS_MIN_FACE = 20   # Tăng kích thước tối thiểu
ABS_MAX_FACE = 300  # Absolute maximum face size
CONF_THRESH_LOW = 0.25  # Tăng ngưỡng cơ bản
MIN_FACE_AREA = 25*25  # Tăng diện tích tối thiểu

# -----------------------------------
# 1. Load .mat and extract HOG features
# -----------------------------------
def load_hog_dataset(pos_path, neg_path):
    mat_pos = scipy.io.loadmat(pos_path)['possamples']
    mat_neg = scipy.io.loadmat(neg_path)['negsamples']
    
    def to_images(mat):
        if mat.ndim == 3:
            h, w, n = mat.shape
            imgs = np.transpose(mat, (2, 0, 1)).astype('float32')
            return imgs, (h, w)
        elif mat.ndim == 2:
            n, feats = mat.shape
            side = int(np.sqrt(feats))
            imgs = mat.reshape(n, side, side).astype('float32')
            return imgs, (side, side)
        else:
            raise ValueError("Unsupported .mat array shape")
    
    pos_imgs, (h_pos, w_pos) = to_images(mat_pos)
    neg_imgs, (h_neg, w_neg) = to_images(mat_neg)
    
    assert (h_pos, w_pos) == (h_neg, w_neg), "Positive/negative sizes differ!"
    H, W = h_pos, w_pos

    def compute_hog(imgs):
        feats = []
        for im in imgs:
            im_norm = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            fd = hog(im_norm, **HOG_PARAMS)
            feats.append(fd)
        return np.array(feats)
    
    X_pos = compute_hog(pos_imgs)
    X_neg = compute_hog(neg_imgs)
    
    X = np.vstack((X_pos, X_neg))
    y = np.hstack((np.ones(len(X_pos)), np.zeros(len(X_neg))))
    return X, y, (H, W)

# -----------------------------------
# 2. Soft-Non Maximum Suppression
# -----------------------------------
def soft_nms(boxes, scores, iou_thr=0.3, sigma=0.5, thresh=0.001):
    if len(boxes) == 0:
        return [], []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    
    indices = np.arange(len(boxes))
    for i in range(len(boxes)):
        max_idx = i
        for j in range(i + 1, len(boxes)):
            if scores[indices[j]] > scores[indices[max_idx]]:
                max_idx = j
        
        indices[i], indices[max_idx] = indices[max_idx], indices[i]
        
        xx1 = np.maximum(x1[indices[i]], x1[indices[i+1:]])
        yy1 = np.maximum(y1[indices[i]], y1[indices[i+1:]])
        xx2 = np.minimum(x2[indices[i]], x2[indices[i+1:]])
        yy2 = np.minimum(y2[indices[i]], y2[indices[i+1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou = intersection / (areas[indices[i]] + areas[indices[i+1:]] - intersection + 1e-10)
        
        weights = np.exp(-(iou * iou) / sigma)
        scores[indices[i+1:]] *= weights
    
    keep = indices[scores[indices] > thresh]
    return boxes[keep], scores[keep]

# -----------------------------------
# 3. Face Detection with Adaptive Parameters
# -----------------------------------
def detect_faces(img, model, scaler, window_size):
    original_shape = img.shape
    img_h, img_w = original_shape[:2]
    
    # Calculate adaptive parameters based on image size
    min_dim = min(img_h, img_w)
    max_dim = max(img_h, img_w)
    
    # Tự động xác định loại ảnh (ít người hay đông người)
    is_crowd_image = max_dim > 200 and min_dim > 200  # Ảnh lớn thường là đông người
    
    # Adaptive face size limits
    if max_dim < 250:  # Small images
        min_face_size = max(ABS_MIN_FACE, int(min_dim * 0.15))
        max_face_size = min(ABS_MAX_FACE, int(max_dim * 0.7))
        scales = [0.95, 1.0, 1.05]  # Reduced scales for small images
        step_size = 10
        nms_thr = 1.4
        conf_thr_factor = 2.5
        sigma = 0.2
    elif is_crowd_image:  # Large images with many faces
        min_face_size = max(ABS_MIN_FACE, int(min_dim * 0.03))
        max_face_size = min(ABS_MAX_FACE, int(max_dim * 0.2))  # Giảm kích thước tối đa
        scales = [0.1, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4]  # Scale tập trung hơn
        step_size = 6
        nms_thr = 0.3  # Giảm ngưỡng NMS cho ảnh đông người
        conf_thr_factor = 0.7  # Giảm hệ số ngưỡng confidence
        sigma = 0.4
    else:  # Medium images (1-5 người)
        min_face_size = max(ABS_MIN_FACE, int(min_dim * 0.05))
        max_face_size = min(ABS_MAX_FACE, int(max_dim * 0.4))
        scales = [0.6, 0.8, 1.0, 1.2]  # Scale tập trung hơn
        step_size = 8
        nms_thr = 0.7
        conf_thr_factor = 1.2  # Tăng hệ số ngưỡng confidence
        sigma = 0.4
    
    print(f"Adaptive params: min_face={min_face_size}px, max_face={max_face_size}px, scales={len(scales)}, step={step_size}, sigma={sigma:.2f}, nms_thr={nms_thr:.2f}, conf_thr_factor={conf_thr_factor:.2f}")
    
    # Resize large images for faster processing
    scale_factor = 1.0
    if max_dim > 1600:
        scale_factor = 1600 / max_dim
        img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)

    ih, iw = gray.shape
    wh, ww = window_size
    all_boxes, all_scores = [], []
    decision_scores = []  # Store all decision scores for analysis

    for scale in scales:
        sw, sh = int(iw * scale), int(ih * scale)
        if sw < ww or sh < wh:
            continue
            
        resized = cv2.resize(gray, (sw, sh))
        adaptive_step = max(4, int(step_size * scale))

        win_w = int(ww * scale)
        win_h = int(wh * scale)
        
        for y in range(0, sh - win_h + 1, adaptive_step):
            for x in range(0, sw - win_w + 1, adaptive_step):
                patch = resized[y:y+win_h, x:x+win_w]
                patch = cv2.resize(patch, (ww, wh))

                fd = hog(patch, **HOG_PARAMS).reshape(1, -1)
                fd = scaler.transform(fd)
                score = model.decision_function(fd)[0]
                decision_scores.append(score)
                
                if score > CONF_THRESH_LOW:
                    ox = int(x / scale)
                    oy = int(y / scale)
                    ow = int(ww / scale)
                    oh = int(wh / scale)
                    
                    # Điều chỉnh tọa độ nếu ảnh đã resize
                    if scale_factor < 1.0:
                        ox = int(ox / scale_factor)
                        oy = int(oy / scale_factor)
                        ow = int(ow / scale_factor)
                        oh = int(oh / scale_factor)
                    
                    # Use adaptive face size limits
                    if (min_face_size <= ow <= max_face_size and
                        0.6 <= ow/oh <= 1.5 and
                        ow*oh >= MIN_FACE_AREA):
                        all_boxes.append([ox, oy, ow, oh])
                        all_scores.append(score)

    if not all_boxes:
        return np.empty((0, 4)), np.array([]), decision_scores

    # Dynamic threshold calculation
    adaptive_threshold = calculate_adaptive_threshold(all_scores, decision_scores)
    adaptive_threshold *= conf_thr_factor
    adaptive_threshold = max(CONF_THRESH_LOW, min(adaptive_threshold, 0.85))
    print(f"Adaptive confidence threshold: {adaptive_threshold:.4f}")

    # Apply adaptive threshold
    filtered_boxes = []
    filtered_scores = []
    for i in range(len(all_scores)):
        if all_scores[i] >= adaptive_threshold:
            filtered_boxes.append(all_boxes[i])
            filtered_scores.append(all_scores[i])
    
    if not filtered_boxes:
        return np.empty((0, 4)), np.array([]), decision_scores

    # Apply Soft-NMS
    boxes, scores = soft_nms(filtered_boxes, filtered_scores, iou_thr=nms_thr, sigma=sigma)
    
    # Cluster detections
    boxes, scores = cluster_detections(boxes, scores, original_shape, img, is_crowd_image)

    return boxes, scores, decision_scores

def calculate_adaptive_threshold(positive_scores, all_scores):
    """Calculate adaptive threshold using score distribution analysis"""
    if not positive_scores:
        return 0.35  # Default threshold
    
    # 1. Calculate basic statistics
    mean_score = np.mean(positive_scores)
    std_score = np.std(positive_scores)
    
    # 2. Kernel Density Estimation for score distribution
    try:
        if len(positive_scores) > 1:
            kde = gaussian_kde(positive_scores)
            x = np.linspace(min(positive_scores), max(positive_scores), 100)
            y = kde(x)
            
            # Find valley between face and non-face clusters
            valleys = []
            for i in range(1, len(y)-1):
                if y[i-1] > y[i] < y[i+1]:
                    valleys.append(x[i])
            
            if valleys:
                valley_thresh = min(valleys)
            else:
                valley_thresh = mean_score - 0.5 * std_score
        else:
            valley_thresh = mean_score - 0.5 * std_score
    except:
        valley_thresh = mean_score - 0.5 * std_score
    
    # 3. Percentile-based threshold
    percentile_thresh = np.percentile(positive_scores, 50)  # Increased from 30
    
    # 4. Combine methods
    adaptive_threshold = max(
        CONF_THRESH_LOW,
        min(mean_score - 0.3 * std_score, valley_thresh, percentile_thresh)
    )
    
    # Ensure threshold is reasonable
    adaptive_threshold = max(CONF_THRESH_LOW, min(adaptive_threshold, 0.85))
    
    return adaptive_threshold

def cluster_detections(boxes, scores, img_shape, img, is_crowd_image=False):
    if len(boxes) == 0:
        return np.empty((0, 4)), np.array([])
    
    centers = np.array([[x + w/2, y + h/2] for x, y, w, h in boxes])
    
    # Adaptive epsilon based on image size
    if is_crowd_image:
        eps = min(img_shape[:2]) / 20  # Tăng epsilon cho ảnh đông người
    else:
        eps = min(img_shape[:2]) / 12  # Giảm epsilon cho ảnh ít người
    
    db = DBSCAN(eps=eps, min_samples=1).fit(centers)
    labels = db.labels_
    
    clusters = defaultdict(list)
    for i, label in enumerate(labels):
        clusters[label].append((boxes[i], scores[i]))
    
    final_boxes = []
    final_scores = []
    
    for label, detections in clusters.items():
        if not detections:
            continue
            
        # Filter by size consistency
        cluster_sizes = [box[2]*box[3] for box, _ in detections]
        median_size = np.median(cluster_sizes)
        
        # Keep only detections within size range
        filtered = [
            (box, score) for box, score in detections 
            if 0.5 * median_size <= box[2]*box[3] <= 2.0 * median_size
        ]
        
        if not filtered:
            continue
            
        # Confidence-based weighting
        weights = np.array([score for _, score in filtered])
        weights /= weights.sum()
        avg_box = np.average([box for box, _ in filtered], axis=0, weights=weights)
        avg_score = np.mean([score for _, score in filtered])
        
        # Validate face
        if validate_face(img, avg_box, avg_score, skip_eye_check=is_crowd_image):
            final_boxes.append(avg_box)
            final_scores.append(avg_score)
    
    return np.array(final_boxes), np.array(final_scores)

# -----------------------------------
# 4. Enhanced Face Validation
# -----------------------------------
def validate_face(img, box, score, skip_eye_check=False):
    x, y, w, h = map(int, box)
    
    # Đảm bảo box nằm trong ảnh
    h_img, w_img = img.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = max(1, min(w, w_img - x))
    h = max(1, min(h, h_img - y))
    
    face_roi = img[y:y+h, x:x+w]
    if face_roi.size == 0 or w <= 5 or h <= 5:
        return False

    # Skip validation for high-confidence detections
    if score > 0.7:
        return True

    gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Minimum size check
    if w < 25 or h < 25:  # Tăng kích thước tối thiểu
        return False

    # Contrast check
    contrast = np.std(gray_face)
    if contrast < 15:  # Tăng ngưỡng tương phản
        return False

    # Aspect ratio check
    aspect_ratio = w / h
    if aspect_ratio < 0.6 or aspect_ratio > 1.5:  # Mở rộng ngưỡng tỷ lệ
        return False

    # Skin color detection
    hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
    skin_pixels = cv2.inRange(hsv, (0, 30, 60), (25, 150, 255))
    skin_ratio = np.sum(skin_pixels > 0) / (w * h)
    if skin_ratio < 0.25:  # Giảm ngưỡng cho ảnh đông người
        return False
        
    # Skip eye check for crowd images or small faces
    if skip_eye_check or w < 50 or h < 50:
        return True

    # Eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if eye_cascade.empty():
        return True  # Skip if cascade not available
    
    eyes = eye_cascade.detectMultiScale(
        gray_face,
        scaleFactor=1.05,
        minNeighbors=3,  # Tăng minNeighbors
        minSize=(int(w * 0.1), int(h * 0.1)),  # Tăng kích thước mắt tối thiểu
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    valid_eyes = 0
    for (ex, ey, ew, eh) in eyes:
        # Kiểm tra vị trí mắt (nên nằm ở nửa trên khuôn mặt)
        if ey < h * 0.6 and (ex > w*0.1 and ex+ew < w*0.9):
            valid_eyes += 1
    
    # Yêu cầu ít nhất 1 mắt hợp lệ
    return valid_eyes >= 1

# -----------------------------------
# 5. Display Results (Improved)
# -----------------------------------
def visualize_results(img, boxes, scores, title="Detection Results"):
    img_display = img.copy()
    if len(img_display.shape) == 2:
        img_display = cv2.cvtColor(img_display, cv2.COLOR_GRAY2BGR)
    
    # Sort by confidence to draw best box last
    sorted_indices = np.argsort(scores)
    
    for i in sorted_indices:
        x, y, w, h = map(int, boxes[i])
        score = scores[i]
        
        # Color based on confidence
        color_intensity = min(255, int(255 * score / max(1, np.max(scores) * 1.5)))
        color = (0, color_intensity, 255 - color_intensity)
        
        cv2.rectangle(img_display, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_display, f"{score:.2f}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Convert to RGB for display
    img_rgb = cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f"{title} - {len(boxes)} faces")
    plt.axis('off')
    plt.show()
    
    return img_display

# -----------------------------------
# 6. Main: Load → Train → Test
# -----------------------------------
if __name__ == "__main__":
    base_dir = r"E:\Project\HTTT\ai-lab\face_recognition"
    pos_mat = os.path.join(base_dir, "possamples.mat")
    neg_mat = os.path.join(base_dir, "negsamples.mat")
    
    print("1) Loading dataset & extracting HOG …")
    X, y, window = load_hog_dataset(pos_mat, neg_mat)
    print(f"   Dataset size: {X.shape}, window: {window}")
    
    print("2) Scaling features & splitting …")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    Xtr, Xva, ytr, yva = train_test_split(Xs, y, test_size=0.2, random_state=42)
    
    print("3) Training SVM with optimized parameters …")
    clf = LinearSVC(C=0.5, max_iter=10000, random_state=42, class_weight='balanced')
    clf.fit(Xtr, ytr)
    
    train_acc = clf.score(Xtr, ytr)
    val_acc = clf.score(Xva, yva)
    print(f"   Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")
    
    print("4) Recognizing faces on test images …")
    test_images = []
    for i in range(1, 3):  # Test on more images
        path = os.path.join(base_dir, f"img{i}.jpg")
        if os.path.exists(path):
            test_images.append(path)
        else:
            # Try with different extensions
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']:
                path = os.path.join(base_dir, f"img{i}{ext}")
                if os.path.exists(path):
                    test_images.append(path)
                    break
    
    for img_path in test_images:
        img = cv2.imread(img_path)
        if img is None:
            print(f"   Could not read {img_path}")
            continue
        
        filename = os.path.basename(img_path)
        print(f"\nProcessing: {filename}")
        start_time = time.time()

        # Detect faces with adaptive parameters
        boxes, scores, _ = detect_faces(img, clf, scaler, window_size=window)
        
        # Filter results using face features with adaptive confidence
        valid_boxes = []
        valid_scores = []
        
        for box, score in zip(boxes, scores):
            if validate_face(img, box, score):
                valid_boxes.append(box)
                valid_scores.append(score)
        
        # Convert to numpy arrays
        valid_boxes = np.array(valid_boxes)
        valid_scores = np.array(valid_scores)
        
        process_time = time.time() - start_time
        print(f"   Processing time: {process_time:.2f}s")
        print(f"   Faces detected: {len(valid_boxes)}")
        
        # Display results
        visualize_results(img, valid_boxes, valid_scores)
        