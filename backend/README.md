# Face-to-Meme Backend

## What This Backend Does

This is a **real-time emotion-based politician meme generator**. It analyzes a user's facial expression from a captured image and returns a matching politician meme based on the detected emotion.

---

## Pipeline Overview

```
User Image → Emotion Detection → Meme Selection → Output Meme
```

### **Stage 1: Emotion Detection**
- Takes a user's face image as input
- Uses **MobileNetV2** deep learning model (`emotion_teller.pth`)
- Trained on FER2013 dataset
- Returns one of 7 emotions: `happy`, `sad`, `angry`, `neutral`, `surprise`, `fear`, `disgust`

### **Stage 2: Meme Selection**
- Takes the detected emotion string
- Randomly selects a matching politician meme from `assets/memes/{emotion}/`
- Copies selected meme to `outputs/final_meme.png` for display

---

## Data Flow

```
1. User captures photo from video feed
   ↓
2. Image → emotion_detector.py → Emotion string ("happy")
   ↓
3. Emotion → politician_selector.py → Random meme selection
   ↓
4. Meme copied to outputs/final_meme.png
   ↓
5. Frontend displays the meme
```

---

## Project Structure

```
backend/
├── main.py                      # FastAPI application entry point
├── requirements.txt             # Python dependencies
│
├── services/                    # Core business logic
│   ├── emotion_detector.py      # Stage 1: Emotion detection
│   ├── politician_selector.py   # Stage 2: Meme selection
│   └── meme_retrieval.py       # Wrapper utilities
│
├── routes/                      # API endpoints
│   ├── capture_routes.py        # POST /capture endpoint
│   └── meme_routes.py          # Legacy routes
│
├── models/                      # ML models
│   └── emotion_teller.pth      # Your emotion detection model
│
├── assets/memes/               # Meme repository (organized by emotion)
│   ├── happy/
│   ├── sad/
│   ├── angry/
│   ├── neutral/
│   ├── surprise/
│   ├── fear/
│   └── disgust/
│
├── outputs/                    # Generated output
│   └── final_meme.png         # Final meme ready for display
│
└── data/
    └── meme_map.csv           # Emotion-to-folder mapping
```

---

## Key Features

✅ **MobileNetV2 Emotion Detection** - Deep learning model trained on FER2013 dataset  
✅ **GPU/CPU Support** - Automatic device detection (CUDA/CPU)  
✅ **High Accuracy** - Custom classifier with dropout and batch normalization  
✅ **Meme Selection** - Randomly picks matching politician memes  
✅ **Output Management** - Copies memes to standard output location  
✅ **Placeholder Mode** - Works without model for testing  
✅ **Error Handling** - Comprehensive error messages  
✅ **Modular Design** - Easy to extend and modify  

---

## Tech Stack

- **Framework:** FastAPI (async Python web framework)
- **Deep Learning:** PyTorch, TorchVision (MobileNetV2 architecture)
- **ML Libraries:** scikit-learn, TensorFlow
- **Image Processing:** Pillow, OpenCV, NumPy
- **Model:** MobileNetV2 with custom 7-class emotion classifier
- **Dependencies:** See `requirements.txt`

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Model
## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your Model
Place your trained MobileNetV2 model file at `models/emotion_teller.pth`  
*(Must be state_dict format with 7-class classifier)*

### 3. Add Meme Images
Add politician meme images to emotion subfolders:
- `assets/memes/happy/politician1.png`
- `assets/memes/sad/politician2.png`
- etc.

### 4. Test the Pipeline
```bash
python test_complete_pipeline.py
```

### 5. Run the Server
```bash
uvicorn main:app --reload
```

---

## Testing

**Test Complete Pipeline:**
```bash
python test_complete_pipeline.py
```

**Test Stage 1 (Emotion Detection):**
```python
from services.emotion_detector import detect_emotion_from_image
emotion = detect_emotion_from_image("test_image.png")
print(emotion)  # Output: "happy"
```

**Test Stage 2 (Meme Selection):**
```python
from services.politician_selector import select_and_output_meme
result = select_and_output_meme("happy")
print(result['output_path'])  # Output: "outputs/final_meme.png"
```

---

## Current Status

| Component | Status |
|-----------|--------|
| Stage 1 - Emotion Detection | ✅ Coded (placeholder until model added) |
| Stage 2 - Meme Selection | ✅ Coded and working |
| Output Management | ✅ Working |
| Pipeline Integration | ✅ Working |
| API Endpoints | 🔄 Placeholder (needs implementation) |

---

## Use Case

An interactive application where users can:
1. Capture their facial expression via webcam
2. Get instant emotion detection
3. Receive a matching politician meme
4. Share or display the meme

Perfect for entertainment, social media engagement, or hackathon demos.

---

## For More Details

See `Function.md` for detailed implementation documentation.
