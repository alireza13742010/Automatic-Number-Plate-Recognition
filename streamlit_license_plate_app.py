import streamlit as st
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2VLForConditionalGeneration
import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
import time
from PIL import Image
import tempfile
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="License Plate Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)


class LicensePlateDetector:
    """License plate detection and OCR with multiple model support."""
    
    def __init__(self):
        self.yolo_model = None
        self.ocr_model = None
        self.ocr_processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_yolo_model(self, model_choice: str, weights_path: str = "best.pt"):
        """Load YOLO model based on user selection."""
        st.info(f"Loading {model_choice}...")
        
        # Map model choices to configurations
        model_configs = {
            "YOLO11 Normal": {"weights": weights_path, "size": 640},
            "YOLO11 XLarge": {"weights": weights_path, "size": 1280},  # Using same weights, different input size
            "YOLO26 Normal": {"weights": weights_path, "size": 640},
            "YOLO26 XLarge": {"weights": weights_path, "size": 1280}
        }
        
        config = model_configs.get(model_choice, model_configs["YOLO11 Normal"])
        
        try:
            self.yolo_model = YOLO(config["weights"])
            self.img_size = config["size"]
            st.success(f"✓ {model_choice} loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Error loading YOLO model: {str(e)}")
            return False
    
    def load_ocr_model(self, ocr_choice: str):
        """Load OCR model based on user selection."""
        st.info(f"Loading {ocr_choice}...")
        
        try:
            if ocr_choice == "GLM-OCR":
                model_path = "zai-org/GLM-OCR"
                self.ocr_processor = AutoProcessor.from_pretrained(model_path)
                
                if self.device == "cuda":
                    self.ocr_model = AutoModelForImageTextToText.from_pretrained(
                        pretrained_model_name_or_path=model_path,
                        torch_dtype=torch.float16,
                        device_map="cuda:0"
                    )
                else:
                    self.ocr_model = AutoModelForImageTextToText.from_pretrained(
                        pretrained_model_name_or_path=model_path,
                        torch_dtype="auto",
                        device_map="cpu"
                    )
                
            elif ocr_choice == "Qwen2.5-VL OCR":
                model_path = "Qwen/Qwen2-VL-7B-Instruct"  # Adjust model name if needed
                self.ocr_processor = AutoProcessor.from_pretrained(model_path)
                
                if self.device == "cuda":
                    self.ocr_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        pretrained_model_name_or_path=model_path,
                        torch_dtype=torch.float16,
                        device_map="cuda:0"
                    )
                else:
                    self.ocr_model = Qwen2VLForConditionalGeneration.from_pretrained(
                        pretrained_model_name_or_path=model_path,
                        torch_dtype="auto",
                        device_map="cpu"
                    )
            
            self.ocr_choice = ocr_choice
            st.success(f"✓ {ocr_choice} loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error loading OCR model: {str(e)}")
            return False
    
    def detect_plates(self, frame: np.ndarray, conf_thresh: float = 0.25):
        """Detect license plates in a frame."""
        if self.yolo_model is None:
            return []
        
        results = self.yolo_model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=conf_thresh,
            device=0 if self.device == "cuda" else "cpu",
            save=False,
            verbose=False,
            half=True if self.device == "cuda" else False
        )
        
        if len(results) == 0:
            return []
        
        res = results[0]
        if not hasattr(res, "boxes") or len(res.boxes) == 0:
            return []
        
        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        scores = res.boxes.conf.cpu().numpy()
        
        detections = []
        for xyxy, cls_id, conf in zip(boxes, classes, scores):
            x1, y1, x2, y2 = xyxy
            detections.append((x1, y1, x2, y2, conf, cls_id))
        
        return detections
    
    def recognize_text_glm(self, plate_image: np.ndarray) -> str:
        """Perform OCR using GLM-OCR."""
        if plate_image.size == 0:
            return ""
        
        try:
            plate_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(plate_rgb)
            
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Text Recognition:"}
                ],
            }]
            
            inputs = self.ocr_processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.ocr_model.device)
            
            inputs.pop("token_type_ids", None)
            
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        generated_ids = self.ocr_model.generate(
                            **inputs,
                            max_new_tokens=512,
                            do_sample=False,
                            num_beams=1
                        )
                else:
                    generated_ids = self.ocr_model.generate(
                        **inputs,
                        max_new_tokens=512,
                        do_sample=False,
                        num_beams=1
                    )
            
            output_text = self.ocr_processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return ' '.join(output_text.strip().split())
            
        except Exception as e:
            st.warning(f"OCR error: {e}")
            return ""
    
    def recognize_text_qwen(self, plate_image: np.ndarray) -> str:
        """Perform OCR using Qwen2.5-VL."""
        if plate_image.size == 0:
            return ""
        
        try:
            plate_rgb = cv2.cvtColor(plate_image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(plate_rgb)
            
            # Qwen2-VL format
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": "Read the text in this license plate image."}
                ]
            }]
            
            text_input = self.ocr_processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.ocr_processor(
                text=[text_input],
                images=[pil_image],
                return_tensors="pt"
            ).to(self.ocr_model.device)
            
            with torch.no_grad():
                if self.device == "cuda":
                    with torch.cuda.amp.autocast():
                        generated_ids = self.ocr_model.generate(
                            **inputs,
                            max_new_tokens=128
                        )
                else:
                    generated_ids = self.ocr_model.generate(
                        **inputs,
                        max_new_tokens=128
                    )
            
            output_text = self.ocr_processor.decode(
                generated_ids[0],
                skip_special_tokens=True
            )
            
            return ' '.join(output_text.strip().split())
            
        except Exception as e:
            st.warning(f"OCR error: {e}")
            return ""
    
    def recognize_text(self, plate_image: np.ndarray) -> str:
        """Route to appropriate OCR method."""
        if self.ocr_choice == "GLM-OCR":
            return self.recognize_text_glm(plate_image)
        else:
            return self.recognize_text_qwen(plate_image)
    
    def process_image(self, image: np.ndarray, conf_thresh: float = 0.25):
        """Process a single image."""
        detections = self.detect_plates(image, conf_thresh)
        output_image = image.copy()
        results = []
        
        for det_idx, detection in enumerate(detections):
            x1, y1, x2, y2, confidence, cls_id = detection
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            plate_crop = image[y1:y2, x1:x2]
            plate_text = self.recognize_text(plate_crop)
            
            if not plate_text:
                plate_text = "Unknown"
            
            results.append({
                'text': plate_text,
                'confidence': float(confidence),
                'bbox': [x1, y1, x2, y2],
                'crop': plate_crop
            })
            
            # Draw bounding box
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255) if confidence > 0.4 else (0, 0, 255)
            thickness = 3 if confidence > 0.5 else 2
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
            
            # Draw text
            text = f"{plate_text} ({confidence:.2f})"
            font_scale = 0.8
            font_thickness = 2
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            padding = 5
            bg_y1 = max(0, y1 - text_height - padding * 2)
            cv2.rectangle(
                output_image,
                (x1, bg_y1),
                (x1 + text_width + padding * 2, y1),
                color,
                -1
            )
            
            cv2.putText(
                output_image,
                text,
                (x1 + padding, y1 - padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                font_thickness
            )
        
        return output_image, results
    
    def process_video(self, video_path: str, conf_thresh: float = 0.25, skip_frames: int = 1):
        """Process a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Cannot open video file")
            return None, []
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create temporary output file
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % skip_frames == 0:
                processed_frame, detections = self.process_image(frame, conf_thresh)
                all_detections.extend(detections)
            else:
                processed_frame = frame
            
            out.write(processed_frame)
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
        
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        return output_path, all_detections


def main():
    st.markdown('<p class="main-header">🚗 License Plate Detection System</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'detector' not in st.session_state:
        st.session_state.detector = LicensePlateDetector()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Device info
        device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
        st.info(f"**Device:** {device}")
        if torch.cuda.is_available():
            st.success(f"**GPU:** {torch.cuda.get_device_name(0)}")
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Detection Model")
        yolo_model = st.selectbox(
            "Select YOLO Model",
            ["YOLO11 Normal", "YOLO11 XLarge", "YOLO26 Normal", "YOLO26 XLarge"],
            help="Choose the YOLO model for license plate detection"
        )
        
        weights_path = st.text_input(
            "YOLO Weights Path",
            value="best.pt",
            help="Path to your trained YOLO weights file"
        )
        
        st.markdown("---")
        
        # OCR selection
        st.subheader("📝 OCR Model")
        ocr_model = st.selectbox(
            "Select OCR Model",
            ["GLM-OCR", "Qwen2.5-VL OCR"],
            help="Choose the OCR model for text recognition"
        )
        
        st.markdown("---")
        
        # Processing settings
        st.subheader("🎯 Processing Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        # Load models button
        if st.button("🚀 Load Models", type="primary", use_container_width=True):
            with st.spinner("Loading models..."):
                yolo_loaded = st.session_state.detector.load_yolo_model(yolo_model, weights_path)
                ocr_loaded = st.session_state.detector.load_ocr_model(ocr_model)
                
                if yolo_loaded and ocr_loaded:
                    st.session_state.models_loaded = True
                    st.balloons()
                else:
                    st.session_state.models_loaded = False
    
    # Main content
    if not st.session_state.get('models_loaded', False):
        st.warning("⚠️ Please load the models from the sidebar to continue")
        st.info("""
        ### Getting Started:
        1. Select your preferred YOLO detection model
        2. Choose an OCR model for text recognition
        3. Set the confidence threshold
        4. Click **Load Models** to initialize
        5. Upload an image or video to process
        """)
        return
    
    # Input selection
    st.markdown('<p class="sub-header">📤 Upload Media</p>', unsafe_allow_html=True)
    
    input_type = st.radio(
        "Select input type:",
        ["Image", "Video"],
        horizontal=True
    )
    
    if input_type == "Image":
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp"],
            help="Upload an image containing license plates"
        )
        
        if uploaded_file is not None:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Process image
            with st.spinner("Processing image..."):
                output_image, detections = st.session_state.detector.process_image(
                    image, conf_threshold
                )
            
            with col2:
                st.subheader("Detected Plates")
                st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Display results
            if detections:
                st.markdown('<p class="sub-header">📊 Detection Results</p>', unsafe_allow_html=True)
                
                for idx, det in enumerate(detections, 1):
                    with st.expander(f"Detection {idx}: {det['text']} (Confidence: {det['confidence']:.2%})"):
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.image(
                                cv2.cvtColor(det['crop'], cv2.COLOR_BGR2RGB),
                                caption="Plate Crop",
                                use_container_width=True
                            )
                        with col_b:
                            st.markdown(f"""
                            **License Plate Text:** `{det['text']}`  
                            **Confidence:** {det['confidence']:.2%}  
                            **Bounding Box:** {det['bbox']}
                            """)
                
                # Summary
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.markdown(f"""
                **Summary:**  
                - Total detections: **{len(detections)}**  
                - Unique plates: **{len(set(d['text'] for d in detections))}**
                """)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No license plates detected in the image")
    
    else:  # Video
        uploaded_file = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv"],
            help="Upload a video containing license plates"
        )
        
        skip_frames = st.slider(
            "Process every N frames (higher = faster)",
            min_value=1,
            max_value=10,
            value=1,
            help="Skip frames for faster processing"
        )
        
        if uploaded_file is not None:
            # Save uploaded video to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            tfile.close()
            
            # Display original video
            st.subheader("Original Video")
            st.video(tfile.name)
            
            # Process button
            if st.button("🎬 Process Video", type="primary"):
                with st.spinner("Processing video... This may take a while."):
                    output_path, detections = st.session_state.detector.process_video(
                        tfile.name, conf_threshold, skip_frames
                    )
                
                if output_path and os.path.exists(output_path):
                    st.success("✓ Video processed successfully!")
                    
                    # Display processed video
                    st.subheader("Processed Video")
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Processed Video",
                            data=f,
                            file_name="detected_plates.mp4",
                            mime="video/mp4"
                        )
                    
                    # Display results
                    if detections:
                        st.markdown('<p class="sub-header">📊 Detection Results</p>', unsafe_allow_html=True)
                        
                        unique_plates = set(d['text'] for d in detections)
                        
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.markdown(f"""
                        **Summary:**  
                        - Total detections: **{len(detections)}**  
                        - Unique plates: **{len(unique_plates)}**
                        """)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Show unique plates
                        st.subheader("Detected License Plates:")
                        for plate in sorted(unique_plates):
                            count = sum(1 for d in detections if d['text'] == plate)
                            st.markdown(f"- **{plate}** (detected {count} times)")
                    else:
                        st.warning("No license plates detected in the video")
                
                # Cleanup
                try:
                    os.unlink(tfile.name)
                    if output_path:
                        os.unlink(output_path)
                except:
                    pass


if __name__ == "__main__":
    main()
