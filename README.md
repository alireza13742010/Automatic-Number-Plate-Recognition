# Automatic-Number-Plate-Recognition
This project provides a complete, user‑friendly License Plate Detection and Recognition System built with Streamlit. It integrates YOLO‑based object detection models for accurate license‑plate localization and supports multiple OCR backends, including GLM‑OCR and Qwen2.5‑VL, to extract text from detected plates with high precision. The application allows users to upload images or videos, automatically detect plates, perform OCR, visualize results, and download processed outputs. It is designed to run efficiently on both CPU and GPU, with automatic device selection and optimized inference pipelines for real‑time or near‑real‑time performance.

The system includes a modern, responsive UI with customizable settings such as model selection, confidence thresholds, and frame‑skipping for faster video processing. Each detection is displayed with bounding boxes, confidence scores, cropped plate previews, and recognized text. The app also provides detailed summaries, including total detections and unique plate counts. This tool is ideal for researchers, developers, and practitioners working on intelligent transportation systems, surveillance, or automated monitoring, offering a flexible and extensible framework for license‑plate recognition tasks. 

# UI for the application
<img src="https://github.com/user-attachments/assets/3a974964-aab4-4b09-95d1-061613abf1eb" width="600" />

# Application Results
<img src="https://github.com/user-attachments/assets/88166bf3-7582-4570-b04d-27420c277a8e" width="400" />
<img src="https://github.com/user-attachments/assets/ecbc41c8-83a4-4a8b-b240-b6056fbee508" width="400" />
<img src="https://github.com/user-attachments/assets/ab0f4882-55e5-4123-b38c-789f9114f614" width="400" />
<img src="https://github.com/user-attachments/assets/e569ce75-a881-457d-866f-fcb2ee13ffdd" width="400" />

# Installation Instructions
Follow these steps to set up and run the application locally:

1. Clone the repository
bash
git clone https://github.com/your-username/Automatic-Number-Plate-Recognition.git
cd Automatic-Number-Plate-Recognition

2. Create and activate a virtual environment (recommended)
bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

3. Install dependencies
bash
pip install -r requirements.txt
Make sure you have PyTorch installed with CUDA support if you plan to run on GPU.
You can install it from: https://pytorch.org/get-started/locally/ (pytorch.org in Bing)

4. Download or place your YOLO weights
Place your trained YOLO weights file (e.g., best.pt) in the project directory.

5. Run the Streamlit app
bash
streamlit run app.py
The application will open in your browser at:
Code
http://localhost:8501
