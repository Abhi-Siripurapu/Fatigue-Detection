import os
from pytube import YouTube
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
import random
import time

def download_video(url):
    print(f"Downloading {url}")
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    if stream is None:
        print(f"No audio stream found for {url}")
        return None

    filename = yt.video_id  # change temp file name to YouTube video ID
    try:
        stream.download(output_path='C:\\Users\\Abhinav\\vf_data', filename=filename)
        raw_video_file = os.path.join('C:\\Users\\Abhinav\\vf_data', filename)
        video_file = raw_video_file + '.mp4'  # explicitly add .mp4 extension

        # Renaming the file to include the .mp4 extension
        os.rename(raw_video_file, video_file)

        time.sleep(2)  # pause execution for 2 seconds

        if not os.path.exists(video_file):
            print(f"Downloaded file not found at {video_file}")
            return None

        return video_file
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None

def process_audio(audio, start, end, label):
    current = start
    while current < end:
        duration = min(random.randint(30, 300), end - current)
        segment = audio.subclip(current, current + duration)
        output_path = f'C:\\Users\\Abhinav\\vf_data\\{label}_{int(current)}_{int(current + duration)}.wav'
        try:
            segment.write_audiofile(output_path)  # using write_audiofile instead of export
        except Exception as e:
            print(f"Error processing audio segment {start}-{end}: {e}")
        current += duration


def process_video(url):
    video_file = download_video(url)
    if video_file is None:
        print(f"Skipping {url} due to download error")
        return

    print(f"Processing audio for {url}")
    audio = AudioFileClip(video_file)  # the video_file variable already points to the file path

    # Skip the non-fatigued and moderately fatigued segments
    process_audio(audio, 15*60, 50*60, 0)
    process_audio(audio, 50*60, 85*60, 0.5)

    # Only process the fatigued segment
    process_audio(audio, 85*60, audio.duration, 1.0)

    audio.close()
    os.remove(video_file)  # delete the video file after processing the audio

process_video('youtube.com/watch?v=NXU_M4030nE')

