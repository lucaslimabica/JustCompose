import time
import cv2 as cv
from just_composing import Camera

class RECer(Camera):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.is_recording = False
        self.last_toggle = 0.0
        self.min_interval = 5.0
        self.rec = None 

    def _draw_rec(self, frame):
        if not self.is_recording:
            return
        h, w  =  frame.shape[:2]
        cv.circle(frame, (w-130,30), 10, (0, 0, 255), -1)
        cv.putText(frame, "REC", (w-110, 38), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv.LINE_AA)

    def _should_toggle(self, hands_by_side):
        if "Left" not in hands_by_side or "Right" not in hands_by_side:
            return False
        return hands_by_side["Left"].gesture == "Thumb_Down" and hands_by_side["Right"].gesture == "Thumb_Down"

    def _toggle_recording(self, frame):
        pass

    def _process_video_stream(self):
        with self.mp_hands.Hands() as hand_detector:
            self.cap = cv.VideoCapture(self.device)
            if not self.cap.isOpened():
                print("No video source :(")
                return
            
            while self.cap.isOpened():
                ret,frame = self.cap.read()
                if not ret:
                    break
                frame = cv.flip(frame,1)
                w,h = self._get_frame_dimensions(frame)

                self.recognizer.process_image(frame, w, h)
                hands_by_side = getattr(self.recognizer, "last_hands_by_side", {})

                if self._should_toggle(hands_by_side):
                    self._toggle_recording(frame)

                if self.is_recording and self.rec:
                    self.rec.write_frame(frame)

                self._draw_rec(frame)
                cv.imshow(self.name,frame)

                key = cv.waitKey(1)
                if key in [27,ord("q"), ord("l")]:
                    break
                if cv.getWindowProperty(self.name,cv.WND_PROP_VISIBLE) < 1:
                    break

            self.cap.release()
            cv.destroyAllWindows()
