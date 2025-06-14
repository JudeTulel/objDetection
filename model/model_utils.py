import onnxruntime as ort
import numpy as np
import cv2

# COCO128 class names
CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

def debug_model_output(prediction):
    """Debug function to understand model output structure"""
    print(f"Prediction type: {type(prediction)}")
    print(f"Prediction length: {len(prediction)}")
    
    for i, pred in enumerate(prediction):
        print(f"Output {i} shape: {pred.shape}")
        print(f"Output {i} dtype: {pred.dtype}")
        print(f"Output {i} min/max: {pred.min():.4f} / {pred.max():.4f}")
        
        # Check if this looks like YOLOv5 output format
        if len(pred.shape) == 3:  # Expected: [batch, detections, features]
            batch, detections, features = pred.shape
            print(f"  -> Batch: {batch}, Detections: {detections}, Features: {features}")
            
            # Sample a few detections to see the structure
            sample_dets = pred[0][:5]  # First 5 detections
            print(f"  -> Sample detections shape: {sample_dets.shape}")
            print(f"  -> Sample detection values:")
            for j, det in enumerate(sample_dets):
                print(f"    Det {j}: x={det[0]:.2f}, y={det[1]:.2f}, w={det[2]:.2f}, h={det[3]:.2f}, conf={det[4]:.4f}")
                if len(det) > 5:
                    print(f"class_confs: {det[5:10]}")  # Show first 5 class confidences
        print("-" * 50)

def load_model():
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession("C:/Users/judet/OneDrive/Desktop/KSA/objectDetection/model/coco128.onnx", providers=providers)
    return session

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # current shape [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def preprocess(img):
    img0 = img.copy()
    img, r, pad = letterbox(img0, new_shape=(640, 640))
    img = img.astype(np.float32)
    img /= 255.0
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img)
    return img, r, pad

def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - inter_area
    return inter_area / (union_area + 1e-6)

def nms(boxes, scores, iou_threshold=0.45):
    idxs = scores.argsort()[::-1]
    keep = []
    while idxs.size > 0:
        i = idxs[0]
        keep.append(i)
        if idxs.size == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep

import numpy as np
import cv2

def postprocess_yolo11(prediction, r, pad, conf_thres=0.25, iou_thres=0.45):
    """
    Postprocess YOLO11 model output
    YOLO11 output format is different from YOLOv5
    """
    # YOLO11 typically outputs [batch, 84, num_detections] format
    # where 84 = 4 (bbox) + 80 (class probabilities)
    # No separate objectness score in YOLO11
    
    outputs = prediction[0]  # Get the first output
    
    # Handle different output shapes
    if len(outputs.shape) == 3:
        if outputs.shape[0] == 1:  # Remove batch dimension if present
            outputs = outputs[0]
    
    print(f"YOLO11 output shape: {outputs.shape}")
    
    # YOLO11 format: [84, num_detections] or [num_detections, 84]
    if outputs.shape[0] == 84:  # [84, num_detections]
        outputs = outputs.T  # Transpose to [num_detections, 84]
    
    print(f"After transpose: {outputs.shape}")
    
    boxes = []
    scores = []
    classes = []
    
    for i, detection in enumerate(outputs):
        # YOLO11 format: [x_center, y_center, width, height, class_prob1, class_prob2, ...]
        x_center, y_center, width, height = detection[:4]
        class_probs = detection[4:]  # 80 class probabilities
        
        # Find the class with highest probability
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]
        
        if confidence < conf_thres:
            continue
            
        # Convert from center coordinates to corner coordinates
        x1 = (x_center - width / 2 - pad[0]) / r
        y1 = (y_center - height / 2 - pad[1]) / r
        x2 = (x_center + width / 2 - pad[0]) / r
        y2 = (y_center + height / 2 - pad[1]) / r
        
        boxes.append([x1, y1, x2, y2])
        scores.append(confidence)
        classes.append(class_id)
    
    if not boxes:
        return np.array([]), np.array([]), np.array([])
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    # Apply NMS
    keep = nms(boxes, scores, iou_threshold=iou_thres)
    
    return boxes[keep], scores[keep], classes[keep]

def postprocess_yolo11_alternative(prediction, r, pad, conf_thres=0.25, iou_thres=0.45):
    """
    Alternative YOLO11 postprocessing in case the first version doesn't work
    This handles the case where YOLO11 might have a different output format
    """
    outputs = prediction[0]
    
    if len(outputs.shape) == 3:
        outputs = outputs[0]  # Remove batch dimension
    
    print(f"Alternative - YOLO11 output shape: {outputs.shape}")
    
    # Try different interpretations
    if outputs.shape[1] == 84:  # [num_detections, 84]
        pass  # Already in correct format
    elif outputs.shape[0] == 84:  # [84, num_detections]
        outputs = outputs.T
    else:
        print(f"Unexpected output shape: {outputs.shape}")
        return np.array([]), np.array([]), np.array([])
    
    boxes = []
    scores = []
    classes = []
    
    for detection in outputs:
        # Extract bounding box and class probabilities
        bbox = detection[:4]  # x, y, w, h
        class_scores = detection[4:84]  # 80 class scores
        
        # Get the best class
        class_id = np.argmax(class_scores)
        confidence = class_scores[class_id]
        
        if confidence < conf_thres:
            continue
        
        # Convert coordinates
        x, y, w, h = bbox
        x1 = (x - w/2 - pad[0]) / r
        y1 = (y - h/2 - pad[1]) / r
        x2 = (x + w/2 - pad[0]) / r
        y2 = (y + h/2 - pad[1]) / r
        
        boxes.append([x1, y1, x2, y2])
        scores.append(confidence)
        classes.append(class_id)
    
    if not boxes:
        return np.array([]), np.array([]), np.array([])
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    
    keep = nms(boxes, scores, iou_threshold=iou_thres)
    return boxes[keep], scores[keep], classes[keep]
def draw_boxes(img, boxes, scores, classes, class_names):
    for box, score, cls in zip(boxes, scores, classes):
        x1, y1, x2, y2 = map(int, box)
        
        # Clip coordinates
        x1 = max(0, min(x1, img.shape[1] - 1))
        y1 = max(0, min(y1, img.shape[0] - 1))
        x2 = max(0, min(x2, img.shape[1] - 1))
        y2 = max(0, min(y2, img.shape[0] - 1))

        # Convert cls to int and validate index
        cls = int(cls)
        if cls < 0 or cls >= len(class_names):
            print(f"Warning: Invalid class index {cls}, skipping detection")
            continue
            
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls]} {score:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    return img
# Updated run_inference function for YOLO11
def run_inference_yolo11(frame, session):
    img, r, pad = preprocess(frame)
    pred = session.run(None, {session.get_inputs()[0].name: img})
    
    # Try the main YOLO11 postprocessing first
    try:
        boxes, scores, classes = postprocess_yolo11(pred, r, pad)
        if len(boxes) == 0:
            # Try alternative if no detections
            boxes, scores, classes = postprocess_yolo11_alternative(pred, r, pad)
    except Exception as e:
        print(f"Error in main postprocessing: {e}")
        boxes, scores, classes = postprocess_yolo11_alternative(pred, r, pad)
    
    frame = draw_boxes(frame, boxes, scores, classes, CLASS_NAMES)
    return frame
