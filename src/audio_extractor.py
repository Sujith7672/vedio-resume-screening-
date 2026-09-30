"""
audio_extractor.py
Extracts audio (.wav) track from a candidate's video resume file.
"""

from moviepy import VideoFileClip
import os


def extract_audio(video_path: str, output_dir: str = "sample_data") -> str:
    """
    Extracts audio from a video file and saves it as a .wav file.

    Args:
        video_path: path to the input video file (.mp4)
        output_dir: directory to save the extracted audio

    Returns:
        path to the extracted .wav audio file
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{filename}.wav")

    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()

    return audio_path


if __name__ == "__main__":
    path = extract_audio("sample_data/candidate1.mp4")
    print(f"Audio extracted to: {path}")
