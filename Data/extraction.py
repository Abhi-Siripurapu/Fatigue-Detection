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


def process_audio(audio, start, end, output_path, label):
    try:
        segment = audio.subclip(start, end)
        if fatigued:
            segment.write_audiofile(output_path)
    except Exception as e:
        print(f"Error processing audio segment {start}-{end}: {str(e)}")


def process_video(url):
    print(f"Downloading {url}")
    yt = YouTube(url)
    vf_data_folder = r'C:\\Users\\Abhinav\\vf_data'
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    if stream is not None:
        video_file = os.path.join(vf_data_folder, 'temp_video_file.mp4')
        stream.download(output_path=video_file)
        audio = AudioFileClip(video_file)
        print(f"Processing audio for {url}")
        t = 0
        while t < audio.duration:
            end_t = min(t + 15*60, audio.duration)
            output_path = os.path.join(vf_data_folder, f"{t}_{end_t}.wav")
            fatigued = t >= 5100
            process_audio(audio, t, end_t, output_path, fatigued)
            t = end_t
    else:
        print(f"Failed to download {url}")


process_video("youtube.com/watch?v=Ff4fRgnuFgQ")