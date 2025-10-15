"""
FastAPI app entry point for emotion-to-politician-meme pipeline.
Provides REST API endpoints for emotion detection and meme generation.
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import shutil
import tempfile
from typing import Optional

from services.emotion_detector import EmotionDetector, detect_emotion_from_image
from services.politician_selector import select_and_output_meme, PoliticianSelector

# Initialize FastAPI app
app = FastAPI(
    title="Face-to-Meme API",
    description="Emotion detection and politician meme generation API",
    version="1.0.0"
)

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize emotion detector (loaded once at startup)
emotion_detector = EmotionDetector()


# Response models
class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    success: bool
    message: str
    model_used: str


class MemeResponse(BaseModel):
    emotion: str
    meme_filename: str
    meme_path: str
    output_path: str
    success: bool
    message: str


class PipelineResponse(BaseModel):
    emotion: str
    confidence: float
    meme_filename: str
    output_path: str
    success: bool
    message: str


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Face-to-Meme API",
        "version": "1.0.0",
        "endpoints": {
            "POST /detect-emotion": "Upload image to detect emotion",
            "POST /get-meme": "Get meme for specific emotion",
            "POST /process": "Complete pipeline - upload image and get meme",
            "GET /emotions": "Get list of available emotions",
            "GET /meme": "Get the final generated meme image"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": emotion_detector.model_loaded,
        "device": str(emotion_detector.device)
    }


@app.get("/emotions")
async def list_emotions():
    """Get list of available emotions with memes"""
    selector = PoliticianSelector()
    emotions = selector.get_available_emotions()
    return {
        "emotions": emotions,
        "count": len(emotions)
    }


@app.post("/detect-emotion", response_model=EmotionResponse)
async def detect_emotion(file: UploadFile = File(...)):
    """
    Detect emotion from uploaded image.
    
    Args:
        file: Image file (PNG, JPG, etc.)
    
    Returns:
        EmotionResponse with detected emotion and confidence
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Save uploaded file temporarily
    temp_file = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Copy uploaded file content to temp file
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Detect emotion
        result = emotion_detector.detect_emotion(temp_path)
        
        return EmotionResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file:
            try:
                Path(temp_path).unlink()
            except:
                pass


@app.post("/get-meme")
async def get_meme(emotion: str):
    """
    Get politician meme for specific emotion.
    
    Args:
        emotion: Emotion string (happy, sad, angry, etc.)
    
    Returns:
        MemeResponse with meme information
    """
    try:
        result = select_and_output_meme(emotion)
        
        if not result['success']:
            raise HTTPException(status_code=404, detail=result['message'])
        
        return MemeResponse(
            emotion=emotion,
            meme_filename=result['meme_filename'],
            meme_path=result['meme_path'],
            output_path=result['output_path'],
            success=result['success'],
            message=result.get('output_message', result['message'])
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error selecting meme: {str(e)}")


@app.post("/process", response_model=PipelineResponse)
async def process_pipeline(file: UploadFile = File(...)):
    """
    Complete pipeline: Upload image → Detect emotion → Get matching meme.
    
    Args:
        file: Image file with face
    
    Returns:
        PipelineResponse with emotion, meme info, and output path
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    temp_file = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Stage 1: Detect emotion
        emotion_result = emotion_detector.detect_emotion(temp_path)
        
        if not emotion_result['success']:
            raise HTTPException(status_code=400, detail=emotion_result['message'])
        
        emotion = emotion_result['emotion']
        confidence = emotion_result['confidence']
        
        # Stage 2: Get matching meme
        meme_result = select_and_output_meme(emotion)
        
        if not meme_result['success']:
            raise HTTPException(status_code=404, detail=meme_result['message'])
        
        return PipelineResponse(
            emotion=emotion,
            confidence=confidence,
            meme_filename=meme_result['meme_filename'],
            output_path=meme_result['output_path'],
            success=True,
            message=f"Detected {emotion} with {confidence:.2%} confidence"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    
    finally:
        # Clean up temp file
        if temp_file:
            try:
                Path(temp_path).unlink()
            except:
                pass


@app.get("/meme")
async def get_final_meme():
    """
    Get the final generated meme image.
    
    Returns:
        Image file (final_meme.png)
    """
    output_path = Path(__file__).parent / "outputs" / "final_meme.png"
    
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="No meme generated yet. Process an image first.")
    
    return FileResponse(
        path=output_path,
        media_type="image/png",
        filename="final_meme.png"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
