import random
from pytube import YouTube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import youtube_dl

def download_video(url):
    yt = YouTube(url)
    # Ensure the video is longer than 2 hours
    if yt.length < 7200:
        print("Video is shorter than 2 hours")
        return None
    # Get the highest quality stream
    stream = yt.streams.get_highest_resolution()
    # Check if a stream is available
    if stream is None:
        print("No available streams for this video")
        return None
    # Download the video
    output_path = stream.download()
    return output_path

def split_video(video_file, start_time, end_time, label):
    clip = VideoFileClip(video_file)
    duration = end_time - start_time
    current_time = start_time
    while current_time < end_time:
        # Randomly select a segment length between 30 seconds and 5 minutes
        segment_length = random.randint(30, 300)
        # Ensure the segment won't extend past the end time
        if current_time + segment_length > end_time:
            segment_length = end_time - current_time
        # Extract the segment
        segment_end = current_time + segment_length
        segment = clip.subclip(current_time, segment_end)
        # Extract the audio and save it
        audio_file = f"{label}_{current_time}_{segment_end}.wav"
        segment.audio.write_audiofile(audio_file)
        # Update the current time
        current_time += segment_length

def process_video(url):
    video_file = download_video(url)
    if video_file:
        split_video(video_file, 15*60, 50*60, '0')
        split_video(video_file, 50*60, 85*60, '0.5')
        split_video(video_file, 85*60, 7200, '1')

process_video('https://www.youtube.com/watch?v=dNrTrx42DGQ')