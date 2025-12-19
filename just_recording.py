from datetime import datetime
import cv2 as cv
from just_composing import *
import soundfile as sf
import numpy as np
import threading
import subprocess
import os

import subprocess
import os

class ScreenRecorder:
    def __init__(self):
        self.writer = None
        self.is_recording = False
        self.last_video_path = ""

    def toggle(self, frame, w, h):
        """switch states"""
        if self.is_recording:
            self.stop()
            return False
        else:
            self.start(w, h)
            return True
    
    def start(self, w, h):
        self.last_video_path = f"temp_video_{datetime.now().strftime('%H%M%S')}.mp4"
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        self.writer = cv.VideoWriter(self.last_video_path, fourcc, 20.0, (w, h))
        self.is_recording = True

    def stop(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.is_recording = False

    def merge_audio(self, audio_path):
        final_file = f"JustCompose_Final_{datetime.now().strftime('%d-%m_%H%M%S')}.mp4"
        
        cmd = f'ffmpeg -y -i {self.last_video_path} -i {audio_path} -c:v copy -c:a aac -shortest {final_file}'
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f"Success making the video: {final_file}")
        except Exception as e:
            print(f"ERROR, maybe the FFmpeg?")
        
class StudioCamera(Camera):
    def __init__(self, name="Just Compose Studio", device=0, capture_mode=None):
        super().__init__(name, device, capture_mode)
        self.is_recording_audio = False 
        self.last_trigger_time = 0
        self.cooldown_seconds = 2.0
        
    def _check_record_trigger(self, results):
        if not results.gestures or len(results.gestures) < 2:
            return False
        thumbs_down_count = 0
        for hand_gestures in results.gestures:
            if hand_gestures[0].category_name == "Thumb_Down":
                thumbs_down_count += 1
        return thumbs_down_count == 2

    def _process_video_stream(self):
        with self.mp_hands.Hands() as hand_detector:
            self.cap = cv.VideoCapture(self.device)
            if not self.cap.isOpened(): return

            dj = self.recognizer.dj
            
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret: break

                frame = cv.flip(frame, 1)
                w, h = self._get_frame_dimensions(frame)
                frame, results = self.recognizer.process_image(frame, w, h)

                if results and self._check_record_trigger(results):
                    now = time.time()
                    if now - self.last_trigger_time > self.cooldown_seconds:
                        self.is_recording_audio = not self.is_recording_audio
                        
                        if self.is_recording_audio:
                            dj.start_recording("myTrack.wav")
                        else:
                            dj.stop_recording()
                            
                        self.last_trigger_time = now

                if self.is_recording_audio:
                    cv.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
                    cv.putText(frame, "AUDIO REC", (w - 178, 35), 
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv.imshow(self.name, frame)
                if cv.waitKey(1) in [27, ord("q")]: break
            
            if self.is_recording_audio:
                dj.stop_recording()
            self.cap.release()
            cv.destroyAllWindows()