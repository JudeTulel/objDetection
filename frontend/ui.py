import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tempfile
import time
from collections import Counter

# Import model utilities
from model.model_utils import load_model, preprocess, draw_boxes, CLASS_NAMES
from model.model_utils import postprocess_yolo11, postprocess_yolo11_alternative, nms

st.set_page_config(page_title="YOLO11 Object Detection", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f8fafc;
        padding-top: 0rem !important;
        margin-top: -2rem !important;
    }
    .block-container {
        padding-top: 1rem !important;
        margin-top: 0rem !important;
    }
    .stApp > header {
        background-color: transparent;
    }
    .stButton>button {background-color: #2563eb; color: white; border-radius: 8px;}
    .stSidebar {background-color: #f1f5f9;}
    .stRadio > div {gap: 1rem;}
    .detection-box {
        background-color: #f0f9ff;
        border: 1px solid #0ea5e9;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .detection-item {
        background-color: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("YOLO11 Object Detection")
st.caption("Detect objects using YOLO11 - supports computer webcam, phone camera, and file uploads")

# Detect if user is on mobile
is_mobile = st.checkbox("📱 I'm using a mobile device", help="Check this if you're on a phone/tablet")

with st.sidebar:
    st.header("Settings")
    st.success("Model: YOLO11n")
    
    if is_mobile:
        mode = st.radio("Select Input Mode", [
            "📸 Phone Camera (Photo)", 
            "🎥 Phone Camera (Video)",
            "📁 Upload Image", 
            "📁 Upload Video"
        ])
    else:
        mode = st.radio("Select Input Mode", [
            "🖥️ Computer Webcam", 
            "📁 Upload Image", 
            "📁 Upload Video", 
            "🌐 Stream URL"
        ])
    
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.25, 0.05)
    iou_threshold = st.slider("IoU Threshold", 0.1, 0.9, 0.45, 0.05)

# Load model
session = load_model()

def run_inference(frame):
    """YOLO11 inference function with detection results"""
    img, r, pad = preprocess(frame)
    pred = session.run(None, {session.get_inputs()[0].name: img})
    
    try:
        boxes, scores, classes = postprocess_yolo11(pred, r, pad, conf_threshold, iou_threshold)
        if len(boxes) == 0:
            boxes, scores, classes = postprocess_yolo11_alternative(pred, r, pad, conf_threshold, iou_threshold)
    except Exception as e:
        print(f"Error in postprocessing: {e}")
        try:
            boxes, scores, classes = postprocess_yolo11_alternative(pred, r, pad, conf_threshold, iou_threshold)
        except:
            return frame, []
    
    frame_with_boxes = draw_boxes(frame, boxes, scores, classes, CLASS_NAMES)
    
    # Create detection results
    detections = []
    for i in range(len(boxes)):
        class_name = CLASS_NAMES[int(classes[i])]
        confidence = float(scores[i])
        detections.append({
            'class': class_name,
            'confidence': confidence,
            'box': boxes[i].tolist() if hasattr(boxes[i], 'tolist') else boxes[i]
        })
    
    return frame_with_boxes, detections

def display_detections(detections, title="🎯 Detection Results"):
    """Display detection results in a formatted way"""
    if not detections:
        st.markdown(f"""
        <div class="detection-box">
            <h4>{title}</h4>
            <p>❌ No objects detected</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Count objects by class
    class_counts = Counter([det['class'] for det in detections])
    
    # Create formatted display
    detection_html = f"""
    <div class="detection-box">
        <h4>{title}</h4>
        <p><strong>Total objects detected: {len(detections)}</strong></p>
    """
    
    # Summary of detected classes
    if class_counts:
        detection_html += "<h5>📊 Object Summary:</h5>"
        for class_name, count in class_counts.most_common():
            detection_html += f"<p>• <strong>{class_name}</strong>: {count} object{'s' if count > 1 else ''}</p>"
    
    # Detailed detections
    detection_html += "<h5>📋 Detailed Detections:</h5>"
    for i, det in enumerate(detections, 1):
        confidence_percent = det['confidence'] * 100
        detection_html += f"""
        <div class="detection-item">
            <strong>#{i}: {det['class']}</strong> - Confidence: {confidence_percent:.1f}%
        </div>
        """
    
    detection_html += "</div>"
    st.markdown(detection_html, unsafe_allow_html=True)

# Handle different input modes
if mode == "📸 Phone Camera (Photo)":
    st.info("📱 Use your phone's camera to take a photo")
    
    # Camera input for phones
    camera_photo = st.camera_input("Take a photo with your phone camera")
    
    if camera_photo is not None:
        # Convert to OpenCV format
        image = Image.open(camera_photo)
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Run inference
        result, detections = run_inference(img_np)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("Detection Results")
            st.image(result, channels="BGR", use_column_width=True)
        
        # Display detection results below
        display_detections(detections)

elif mode == "🎥 Phone Camera (Video)":
    st.info("📱 Record a video with your phone camera")
    st.markdown("""
    **Instructions:**
    1. Click 'Browse files' below
    2. Choose 'Camera' or 'Video' when prompted
    3. Record your video
    4. Upload the recorded video
    """)
    
    uploaded_video = st.file_uploader(
        "Record or upload a video", 
        type=["mp4", "avi", "mov", "webm"],
        help="On mobile: tap to record with camera, or select existing video"
    )
    
    if uploaded_video:
        # Process video
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        tfile.close()
        
        cap = cv2.VideoCapture(tfile.name)
        
        # Video display container
        st.subheader("🎥 Video Processing")
        stframe = st.empty()
        
        # Detection results container
        detection_placeholder = st.empty()
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 5th frame for speed
            if frame_count % 5 == 0:
                result_frame, detections = run_inference(frame)
                stframe.image(result_frame, channels="BGR")
                all_detections.extend(detections)
                
                # Update detection display in real-time
                if detections:
                    with detection_placeholder.container():
                        display_detections(detections, f"🎯 Frame {frame_count} Detections")
                
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
            
        cap.release()
        st.success("Video processing complete!")
        
        # Show final summary
        if all_detections:
            st.subheader("📊 Video Summary")
            display_detections(all_detections, "🎯 All Video Detections Summary")

elif mode == "🖥️ Computer Webcam":
    st.info("🖥️ Using your computer's webcam")
    
    # Video display
    st.subheader("🖥️ Webcam Feed")
    stframe = st.empty()
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_btn = st.button("▶️ Start Webcam", type="primary")
    with col2:
        stop_btn = st.button("⏹️ Stop Webcam")
    
    if start_btn and not stop_btn:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not access webcam. Please check your camera permissions.")
        else:
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break
                    
                # Run inference
                result_frame, detections = run_inference(frame)
                stframe.image(result_frame, channels="BGR")
                
                # Small delay to prevent overwhelming the browser
                time.sleep(0.1)
                
            cap.release()

elif mode in ["📁 Upload Image", "📁 Upload Image"]:
    uploaded_file = st.file_uploader(
        "Choose an image", 
        type=["jpg", "jpeg", "png", "webp"],
        help="Select an image file from your device"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        result, detections = run_inference(img_np)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        with col2:
            st.subheader("Detection Results")
            st.image(result, channels="BGR", use_column_width=True)
        
        # Display detection results below
        display_detections(detections)

elif mode in ["📁 Upload Video", "📁 Upload Video"]:
    uploaded_video = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "webm"])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        # Video display
        st.subheader("🎥 Video Processing")
        stframe = st.empty()
        
        # Detection display
        detection_placeholder = st.empty()
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)
        
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % 3 == 0:  # Process every 3rd frame
                result_frame, detections = run_inference(frame)
                stframe.image(result_frame, channels="BGR")
                all_detections.extend(detections)
                
                # Update detection display
                if detections:
                    with detection_placeholder.container():
                        display_detections(detections, f"🎯 Frame {frame_count} Detections")
                
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
            
        cap.release()
        st.success("Video processing complete!")
        
        # Show final summary
        if all_detections:
            st.subheader("📊 Video Summary")
            display_detections(all_detections, "🎯 All Video Detections Summary")

elif mode == "🌐 Stream URL":
    stream_url = st.text_input("Enter Stream URL (RTSP/HTTP)")
    
    if stream_url:
        # Stream display
        st.subheader("🌐 Live Stream")
        stframe = st.empty()
        
        # Stream controls
        col1, col2 = st.columns(2)
        with col1:
            start_stream = st.button("▶️ Start Stream", type="primary")
        with col2:
            stop_stream = st.button("⏹️ Stop Stream")
        
        if start_stream and not stop_stream:
            cap = cv2.VideoCapture(stream_url)
            
            if not cap.isOpened():
                st.error("Could not connect to stream. Please check the URL.")
            else:
                while cap.isOpened() and not stop_stream:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Could not read from stream")
                        break
                        
                    result_frame, detections = run_inference(frame)
                    stframe.image(result_frame, channels="BGR")
                    
                    # Small delay
                    time.sleep(0.1)
                    
                cap.release()

# Add helpful tips
with st.expander("💡 Tips for Best Results"):
    st.markdown("""
    **For Phone Users:**
    - Use 📸 Phone Camera (Photo) for single image detection
    - Use 🎥 Phone Camera (Video) to record and analyze videos
    - Ensure good lighting for better detection accuracy
    - Hold the phone steady when recording
    
    **For Computer Users:**
    - Make sure your webcam permissions are enabled
    - Use good lighting for better detection
    - The app processes frames in real-time (detection results not shown for performance)
    - Stream processing focuses on real-time video display
    
    **Understanding Detection Results:**
    - Object Summary shows count of each detected class
    - Detailed Detections show individual objects with confidence scores
    - Higher confidence scores indicate more certain detections
    - Video processing shows both frame-by-frame and summary results
    
    **General Tips:**
    - Lower confidence threshold to detect more objects
    - Higher confidence threshold for more precise detection
    - Adjust IoU threshold to control overlapping detections
    """)

# Add detection legend
with st.expander("🎯 Detection Classes"):
    st.markdown("**YOLO11 can detect the following object classes:**")
    
    # Display class names in columns
    col1, col2, col3, col4 = st.columns(4)
    
    for i, class_name in enumerate(CLASS_NAMES):
        if i % 4 == 0:
            col1.write(f"• {class_name}")
        elif i % 4 == 1:
            col2.write(f"• {class_name}")
        elif i % 4 == 2:
            col3.write(f"• {class_name}")
        else:
            col4.write(f"• {class_name}")