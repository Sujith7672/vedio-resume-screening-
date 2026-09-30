"""
speech_to_text.py
Transcribes audio to text using OpenAI Whisper.
"""

import whisper


def transcribe_audio(audio_path: str, model_size: str = "base") -> dict:
    """
    Transcribes an audio file to text using Whisper.

    Args:
        audio_path: path to the .wav audio file
        model_size: whisper model size (tiny, base, small, medium, large)

    Returns:
        dict with 'text' (full transcript) and 'segments' (timestamped chunks)
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path)

    return {
        "text": result["text"],
        "segments": result["segments"],
    }


def count_filler_words(transcript: str) -> int:
    """Counts common filler words used as a simple fluency signal."""
    fillers = ["um", "uh", "like", "you know", "basically", "actually"]
    text_lower = transcript.lower()
    return sum(text_lower.count(f) for f in fillers)


if __name__ == "__main__":
    result = transcribe_audio("sample_data/candidate1.wav")
    print(result["text"])
    print("Filler word count:", count_filler_words(result["text"]))
