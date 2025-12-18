from moviepy import VideoFileClip
import os

videos = [
    "./assets/drum01.mp4",
    "./assets/drum02.mp4",
    "./assets/drum03.mp4",
    "./assets/syhnt01.mp4",
    "./assets/syhnt02.mp4",
    "./assets/syhnt03.mp4",
    "./assets/guitar01.mp4",
    "./assets/guitar02.mp4",
    "./assets/guitar03.mp4",
]

for path in videos:
    video = VideoFileClip(path)
    mp3_path = os.path.splitext(path)[0] + ".mp3"
    video.audio.write_audiofile(mp3_path)
    print(f"Extract: {mp3_path}")
