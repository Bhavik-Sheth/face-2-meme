# Face-to-Meme Backend

## Project Overview: Emotion-Based Politician Meme Generator

This is a **real-time emotion detection system** that captures a user's facial expression from a live video feed and returns a matching politician meme based on the detected emotion.

### Pipeline Architecture

**3-Stage ML Pipeline:**

1. **Emotion Detection Stage** (`services/emotion_detector.py`)
   - Uses **DeepFace** library to analyze facial expressions from captured video frames
   - Detects emotions: happy, sad, angry, neutral, surprise, fear, disgust

2. **Politician Selection Stage** (`services/politician_selector.py`)
   - Custom ML model (stored in `models/` folder)
   - Maps detected emotions to specific politicians
   - Trained to associate emotional states with politician personas

3. **Meme Retrieval Stage** (`services/meme_retrieval.py`)
   - Queries `data/meme_map.csv` to find matching memes
   - CSV maps: `emotion → politician → meme_filename`
   - Fetches the corresponding meme image from `assets/memes/`

### API Structure (FastAPI)

**Endpoints:**
- **POST `/capture`** (`routes/capture_routes.py`) — Main endpoint that accepts a video frame, processes it through the pipeline, and returns a politician meme
- **GET `/`** (`main.py`) — Health check/root endpoint

### Data Flow
```
User's Face (video frame) 
    ↓
POST /capture endpoint
    ↓
DeepFace emotion detection
    ↓
ML model selects politician
    ↓
meme_map.csv lookup
    ↓
Returns politician meme image
```

### Tech Stack
- **Backend Framework:** FastAPI (async Python web framework)
- **Emotion Recognition:** DeepFace (pre-trained deep learning model)
- **ML Models:** Custom classifier (scikit-learn/TensorFlow)
- **Image Processing:** Pillow, OpenCV
- **Data Handling:** Pandas for CSV operations

### Key Assets
- **`assets/memes/`** — Repository of politician meme images
- **`data/meme_map.csv`** — Emotion-to-politician-to-meme mapping database
- **`models/`** — Trained ML models for politician selection
- **`outputs/`** — Temporary storage for processed results

### Use Case
This is a **hackathon project** that creates an interactive, humorous application where users can see which politician meme matches their current emotional state—designed for entertainment or social media engagement purposes.

The project demonstrates integration of:
- Computer vision (face detection)
- Emotion AI (sentiment analysis)
- Machine learning (classification)
- Web APIs (FastAPI)
- Real-time processing
