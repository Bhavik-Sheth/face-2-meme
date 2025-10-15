"""
Stage 2: Politician/Meme Selector
Takes emotion from stage 1 and returns a matching politician meme path.
Supports outputting/displaying the selected meme image.
"""
import os
import random
import shutil
from pathlib import Path
from PIL import Image


class PoliticianSelector:
    """Selects a politician meme based on detected emotion."""
    
    def __init__(self, memes_base_path: str = None):
        """Initialize the selector with the base path to memes folder."""
        if memes_base_path is None:
            # Default to backend/assets/memes
            current_dir = Path(__file__).parent.parent
            self.memes_base_path = current_dir / "assets" / "memes"
        else:
            self.memes_base_path = Path(memes_base_path)
        
        # Supported emotions (matching folder names)
        self.supported_emotions = [
            "happy", "sad", "angry", "neutral", 
            "surprise", "fear", "disgust"
        ]
    
    def select_meme(self, emotion: str) -> dict:
        """
        Select a random politician meme for the given emotion.
        
        Args:
            emotion (str): Detected emotion from stage 1 (e.g., "happy", "angry")
        
        Returns:
            dict: {
                "emotion": str,
                "meme_path": str (absolute path to meme image),
                "meme_filename": str,
                "success": bool,
                "message": str
            }
        """
        # Normalize emotion to lowercase
        emotion = emotion.lower().strip()
        
        # Validate emotion
        if emotion not in self.supported_emotions:
            return {
                "emotion": emotion,
                "meme_path": None,
                "meme_filename": None,
                "success": False,
                "message": f"Unsupported emotion: {emotion}. Supported: {self.supported_emotions}"
            }
        
        # Build path to emotion subfolder
        emotion_folder = self.memes_base_path / emotion
        
        # Check if folder exists
        if not emotion_folder.exists():
            return {
                "emotion": emotion,
                "meme_path": None,
                "meme_filename": None,
                "success": False,
                "message": f"Emotion folder not found: {emotion_folder}"
            }
        
        # Get all image files in the emotion folder
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        meme_files = [
            f for f in os.listdir(emotion_folder)
            if os.path.isfile(emotion_folder / f) 
            and any(f.lower().endswith(ext) for ext in image_extensions)
        ]
        
        # Check if there are any memes available
        if not meme_files:
            return {
                "emotion": emotion,
                "meme_path": None,
                "meme_filename": None,
                "success": False,
                "message": f"No meme images found in {emotion_folder}"
            }
        
        # Randomly select a meme
        selected_meme = random.choice(meme_files)
        meme_path = emotion_folder / selected_meme
        
        return {
            "emotion": emotion,
            "meme_path": str(meme_path),
            "meme_filename": selected_meme,
            "success": True,
            "message": "Meme selected successfully"
        }
    
    def get_available_emotions(self) -> list:
        """Return list of emotions that have memes available."""
        available = []
        for emotion in self.supported_emotions:
            emotion_folder = self.memes_base_path / emotion
            if emotion_folder.exists():
                # Check if folder has any image files
                image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".webp"]
                meme_files = [
                    f for f in os.listdir(emotion_folder)
                    if os.path.isfile(emotion_folder / f) 
                    and any(f.lower().endswith(ext) for ext in image_extensions)
                ]
                if meme_files:
                    available.append(emotion)
        return available
    
    def output_meme(self, meme_result: dict, output_path: str = None) -> dict:
        """
        Copy selected meme to output folder for display/serving.
        
        Args:
            meme_result (dict): Result from select_meme()
            output_path (str): Optional custom output path. 
                             If None, uses backend/outputs/final_meme.jpg
        
        Returns:
            dict: {
                "success": bool,
                "output_path": str,
                "original_path": str,
                "message": str
            }
        """
        if not meme_result.get('success', False):
            return {
                "success": False,
                "output_path": None,
                "original_path": None,
                "message": "Cannot output meme - selection failed"
            }
        
        source_path = Path(meme_result['meme_path'])
        
        # Determine output path
        if output_path is None:
            outputs_dir = Path(__file__).parent.parent / "outputs"
            outputs_dir.mkdir(exist_ok=True)
            # Keep original extension
            extension = source_path.suffix
            output_path = outputs_dir / f"final_meme{extension}"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Copy meme to output location
            shutil.copy2(source_path, output_path)
            
            return {
                "success": True,
                "output_path": str(output_path),
                "original_path": str(source_path),
                "message": f"Meme output to {output_path}"
            }
        except Exception as e:
            return {
                "success": False,
                "output_path": None,
                "original_path": str(source_path),
                "message": f"Failed to output meme: {str(e)}"
            }
    
    def display_meme(self, meme_result: dict) -> dict:
        """
        Display the selected meme image using PIL.
        
        Args:
            meme_result (dict): Result from select_meme()
        
        Returns:
            dict: {
                "success": bool,
                "message": str,
                "image_size": tuple (width, height)
            }
        """
        if not meme_result.get('success', False):
            return {
                "success": False,
                "message": "Cannot display meme - selection failed",
                "image_size": None
            }
        
        try:
            meme_path = Path(meme_result['meme_path'])
            img = Image.open(meme_path)
            
            # Display the image (opens in default image viewer)
            img.show()
            
            return {
                "success": True,
                "message": f"Displaying meme: {meme_result['meme_filename']}",
                "image_size": img.size
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to display meme: {str(e)}",
                "image_size": None
            }


# Convenience functions for quick usage
def select_politician_meme(emotion: str) -> dict:
    """
    Quick function to select a meme for a given emotion.
    
    Args:
        emotion (str): Detected emotion from stage 1
    
    Returns:
        dict: Meme selection result
    """
    selector = PoliticianSelector()
    return selector.select_meme(emotion)


def select_and_output_meme(emotion: str, output_path: str = None) -> dict:
    """
    Select a meme and copy it to output folder in one step.
    
    Args:
        emotion (str): Detected emotion from stage 1
        output_path (str): Optional custom output path
    
    Returns:
        dict: Combined result with selection and output info
    """
    selector = PoliticianSelector()
    
    # Select meme
    selection_result = selector.select_meme(emotion)
    
    if not selection_result['success']:
        return selection_result
    
    # Output meme
    output_result = selector.output_meme(selection_result, output_path)
    
    # Combine results
    return {
        **selection_result,
        "output_path": output_result.get('output_path'),
        "output_success": output_result['success'],
        "output_message": output_result['message']
    }


def select_and_display_meme(emotion: str) -> dict:
    """
    Select a meme and display it in one step.
    
    Args:
        emotion (str): Detected emotion from stage 1
    
    Returns:
        dict: Combined result with selection and display info
    """
    selector = PoliticianSelector()
    
    # Select meme
    selection_result = selector.select_meme(emotion)
    
    if not selection_result['success']:
        return selection_result
    
    # Display meme
    display_result = selector.display_meme(selection_result)
    
    # Combine results
    return {
        **selection_result,
        "display_success": display_result['success'],
        "display_message": display_result['message'],
        "image_size": display_result.get('image_size')
    }
