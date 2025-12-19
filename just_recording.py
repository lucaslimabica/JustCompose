import os
import time

import cv2 as cv
import sounddevice as sd
import soundfile as sf
from moviepy import VideoFileClip, AudioFileClip

from just_composing import Camera


class RecordingManager:
    def __init__(self, out_dir="recordings", fps=30, video_size=(720, 720), sr=48000):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.fps = fps
        self.video_size = video_size
        self.sr = sr

        self.recording = False
        self._vw = None
        self._audio_stream = None
        self._audio_file = None
        self._base_path = None
        self._start_t = 0.0

    def _default_loopback_device(self):
        hostapis = sd.query_hostapis()
        wasapi_ids = [i for i, h in enumerate(hostapis) if "WASAPI" in h["name"].upper()]
        if wasapi_ids:
            wid = wasapi_ids[0]
            default_out = hostapis[wid]["default_output_device"]
            if default_out is not None:
                return default_out
        return None

    def start(self):
        if self.recording:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._base_path = os.path.join(self.out_dir, f"take_{ts}")

        mp4_path = self._base_path + ".mp4"
        wav_path = self._base_path + ".wav"

        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        self._vw = cv.VideoWriter(mp4_path, fourcc, self.fps, self.video_size)

        dev = self._default_loopback_device()
        self._audio_file = sf.SoundFile(
            wav_path,
            mode="w",
            samplerate=self.sr,
            channels=2,
            subtype="PCM_16",
        )

        def callback(indata, frames, time_info, status):
            self._audio_file.write(indata.copy())

        extra = {}
        try:
            extra["extra_settings"] = sd.WasapiSettings(loopback=True)
        except Exception:
            extra = {}

        self._audio_stream = sd.InputStream(
            samplerate=self.sr,
            device=dev,
            channels=2,
            callback=callback,
            **extra,
        )
        self._audio_stream.start()

        self.recording = True
        self._start_t = time.time()

    def write_frame(self, frame_bgr):
        if (not self.recording) or (self._vw is None):
            return
        f = cv.resize(frame_bgr, self.video_size, interpolation=cv.INTER_AREA)
        self._vw.write(f)

    def stop(self):
        if not self.recording:
            return None
        self.recording = False

        if self._audio_stream:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None

        if self._audio_file:
            self._audio_file.close()
            self._audio_file = None

        if self._vw:
            self._vw.release()
            self._vw = None

        mp4_path = self._base_path + ".mp4"
        wav_path = self._base_path + ".wav"
        merged_mp4 = self._base_path + "_AV.mp4"

        v = VideoFileClip(mp4_path)
        a = AudioFileClip(wav_path)
        v = v.with_audio(a)
        v.write_videofile(
            merged_mp4,
            codec="libx264",
            audio_codec="aac",
            verbose=False,
            logger=None,
        )

        return merged_mp4

    def elapsed(self):
        return (time.time() - self._start_t) if self.recording else 0.0


class RECer(Camera):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_recording = False
        self.last_toggle = 0.0
        self.min_interval = 5.0
        self.rec = None
        self.toggle_armed = True

    def _draw_rec(self, frame):
        if not self.is_recording:
            return
        h, w = frame.shape[:2]
        cv.circle(frame, (w - 130, 30), 10, (0, 0, 255), -1)
        cv.putText(
            frame,
            "REC",
            (w - 110, 38),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
            cv.LINE_AA,
        )

    def _should_toggle(self, hands_by_side):
        if ("Left" not in hands_by_side) or ("Right" not in hands_by_side):
            return False
        return (
            hands_by_side["Left"].gesture == "Thumb_Down"
            and hands_by_side["Right"].gesture == "Thumb_Down"
        )

    def _toggle_recording(self, frame):
        now = time.time()
        if now - self.last_toggle < self.min_interval:
            return
        self.last_toggle = now

        if not self.is_recording:
            if self.rec is None:
                h, w = frame.shape[:2]
                self.rec = RecordingManager(out_dir="recordings", fps=30, video_size=(w, h), sr=48000)
            self.rec.start()
            self.is_recording = True
            print("[REC] start")
        else:
            merged_mp4 = self.rec.stop()
            self.is_recording = False
            print(f"[REC] stop -> {merged_mp4}")

    def _process_video_stream(self):
        with self.mp_hands.Hands() as hand_detector:
            self.cap = cv.VideoCapture(self.device)
            if not self.cap.isOpened():
                print("No video source :(")
                return

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv.flip(frame, 1)
                w, h = self._get_frame_dimensions(frame)

                self.recognizer.process_image(frame, w, h)
                hands_by_side = getattr(self.recognizer, "last_hands_by_side", {})

                trigger = self._should_toggle(hands_by_side)
                if trigger and self.toggle_armed:
                    self._toggle_recording(frame)
                    self.toggle_armed = False
                elif not trigger:
                    self.toggle_armed = True

                if self.is_recording and self.rec:
                    self.rec.write_frame(frame)

                self._draw_rec(frame)
                cv.imshow(self.name, frame)

                key = cv.waitKey(1)
                if key in [27, ord("q"), ord("l")]:
                    break
                if cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) < 1:
                    break

            self.cap.release()
            cv.destroyAllWindows()
