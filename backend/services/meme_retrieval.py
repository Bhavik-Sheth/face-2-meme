"""
Meme Retrieval Service - Wrapper around politician_selector
Provides backward compatibility and additional utility functions.
"""
from .politician_selector import (
    PoliticianSelector, 
    select_politician_meme,
    select_and_output_meme,
    select_and_display_meme
)


def get_meme_for_emotion(emotion: str, output: bool = False, display: bool = False) -> dict:
    """
    Retrieve a politician meme for the given emotion.
    
    Args:
        emotion (str): Detected emotion (happy, sad, angry, etc.)
        output (bool): If True, copy meme to outputs folder
        display (bool): If True, display meme in image viewer
    
    Returns:
        dict: Meme selection result with path and metadata
    """
    if output:
        return select_and_output_meme(emotion)
    elif display:
        return select_and_display_meme(emotion)
    else:
        return select_politician_meme(emotion)


def get_available_emotions() -> list:
    """Get list of emotions that have memes available."""
    selector = PoliticianSelector()
    return selector.get_available_emotions()


def output_meme_to_folder(emotion: str, output_path: str = None) -> dict:
    """
    Select a meme and output it to specified location.
    
    Args:
        emotion (str): Detected emotion
        output_path (str): Custom output path (optional)
    
    Returns:
        dict: Result with output path information
    """
    return select_and_output_meme(emotion, output_path)
