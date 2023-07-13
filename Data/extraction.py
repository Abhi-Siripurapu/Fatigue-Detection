import os
from pytube import YouTube
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
import random

def download_video(url):
    print(f"Downloading {url}")
    yt = YouTube(url)
    stream = yt.streams.get_highest_resolution()
    if stream is None:
        print(f"No stream available for {url}")
        return None
    out_dir = 'C:\\Users\\Abhinav\\vf_data'
    video_file = stream.download(output_path=out_dir)
    base, ext = os.path.splitext(video_file)
    if ext != ".mp4":
        new_video_file = base + ".mp4"
        os.rename(video_file, new_video_file)
        video_file = new_video_file
    return video_file


def process_audio(audio, start, end, output_path, fatigued):
    end = float(end)
    try:
        segment = audio.subclip(start, end)
        if fatigued:
            segment.write_audiofile(output_path)
    except Exception as e:
        print(f"Error processing audio segment {start}-{end}: {str(e)}")

def sanitize_title(title):
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        if char != ',':
            title = title.replace(char, '_')
    return title



def process_video(url):
    print(f"Downloading {url}")
    yt = YouTube(url)
    stream = yt.streams.get_audio_only()
    if stream is None:
        print(f"No stream available for {url}")
        return None
    stream.download(output_path="C:\\Users\\Abhinav\\vf_data")
    filename = stream.default_filename
        
    output_path = os.path.join("C:\\Users\\Abhinav\\vf_data", filename)
    audio = AudioFileClip(output_path)

    

    process_audio(audio, 85*60, output_path, audio.duration - 200, 1)  # Use audio.duration instead of yt.length/60
    audio.close()  # close the audio clip before removing the file
    os.remove(output_path)  # remove the downloaded file



process_video("youtube.com/watch?v=Ff4fRgnuFgQ")