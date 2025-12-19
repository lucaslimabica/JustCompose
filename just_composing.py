# Responsible Camera for Motion Capture
import cv2 as cv
import mediapipe as mp
import pygame
import database_manager
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import pprint
import time
import fluidsynth
from threading import Timer


class Camera:
    """
    Main camera handler for motion capture and gesture recognition.
    This class is responsible for:
    - Capturing frames from a video source (webcam or image file)
    - Running MediaPipe Hands on each frame
    - Drawing landmarks, labels, and bounding boxes
    - Recognizing gestures based on conditions stored in the database
    """
    
    _TOLERANCE_THRESHOLD = 0.02  # Small tolerance for landmark comparisons
    _CAPTURE_MODES = ["landmarks", "landmarks_coords", "bounding_box"]
    
    def __init__(self, name="Just Compose Beta", device=0, capture_mode=None):
        """
        Initialize the Camera

        Args:
            name (str, optional):
                Window title used by OpenCV when displaying the frames.
                Defaults to "Just Compose Beta".
            device (int | str, optional):
                Capture source.
                - If `int`, it is treated as an OpenCV camera index (0 for default webcam).
                - If `str` and the path ends with a compatible extension
                  ('.jpg', '.jpeg', '.png'), it is treated as an image file path.
                Defaults to 0.
            capture_mode (str | None, optional):
                Controls which text overlays are drawn on top of the detected landmarks.
                Supported values:
                    - "landmarks"         → draw only the landmark indices
                    - "landmarks_coords"  → draw landmark indices + normalized coordinates
                    - "bounding_box"      → draw the bounding box around the hand
                    - None                → do not draw any landmark labels
                Defaults to None.
        """

        # MediaPipe structure
        self.mp_hands = mp.solutions.hands 
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Class attributes
        self.name = name
        self.device = device
        self.compatible_file_types = ('.jpg', '.jpeg', '.png')
        self.capture_mode = capture_mode
        self.recognizer = HandSpeller(running_mode=vision.RunningMode.IMAGE)
    
    def capture(self):
        """
        Initializes the motion capture process based on the configured device.
        
        This method acts as a router:
            - If device is a camera index (int), it delegates to real-time video processing.
            - If device is an image file path (str), it delegates to static image processing.
        
        The delegated process blocks the thread until the window is closed 
        or a termination key (ESC, 'q', or 'l') is pressed.
        """
        if self.device == 0:        
            self._process_video_stream()
        elif isinstance(self.device, str) and self.device.endswith(self.compatible_file_types):
            self._process_image()

    def _process_image(self):
        """
        Loads a static image from self.device and performs a single pass of
        hand detection and gesture recognition.
        
        Behavior:
            - Loads the image using OpenCV.
            - Runs MediaPipe Hands in static_image_mode=True once.
            - Draws landmarks and recognized gestures.
            
        The resulting image is displayed until a termination key is pressed 
        (ESC, 'q', 'l', or window close).
        """       
        image = cv.imread(self.device)
        w, h = self._get_frame_dimensions(image)
        self.recognizer.process_image(image, w, h)
        with self.mp_hands.Hands(static_image_mode=True) as hand_detector:
            results = hand_detector.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                self._capture_hands(image, results)
            cv.imshow(self.name, image)
            
            while True:
                key = cv.waitKey(1) 
                # Keyboard LEGACY
                if key in [27, ord("q"), ord("l")]:
                    break
                if cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) < 1:
                    break
                
        cv.destroyAllWindows()
            
    def _process_video_stream(self):
        """
        Starts the real-time video capture loop for live motion detection.
        """
        with self.mp_hands.Hands() as hand_detector:
            self.cap = cv.VideoCapture(self.device)
            if not self.cap.isOpened(): 
                print("No video source :(")
                return

            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:  # frame not captured
                    print("The video has no frames")
                    break

                # Mirror the frame
                frame = cv.flip(frame, 1)

                w, h = self._get_frame_dimensions(frame)
                self.recognizer.process_image(frame, w, h)

                results = hand_detector.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:  # Avoid None for the drawing func
                    self._capture_hands(frame, results)

                cv.imshow(self.name, frame)

                key = cv.waitKey(1)
                # Keyboard LEGACY
                if key in [27, ord("q"), ord("l")]:
                    break
                if cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) < 1:
                    break

            # Break of the loop -> Release resources
            self.cap.release()
            cv.destroyAllWindows()

                    
    def _capture_hands(self, image, results, draw=False):
        """
        Draw hand landmarks, connections, labels, and recognized gestures
        Args:
            image (numpy.ndarray):
                Current frame (BGR) where the landmarks and overlays will be drawn.
            results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                The result object returned from `mp_hands.Hands.process(...)`,
                containing `multi_hand_landmarks` and `multi_handedness`.
            draw (bool, optional):
                If True, draws the landmarks and connections on the image. Defaults to False.

        Behavior:
            - Iterates over each detected hand and its handedness.
            - Draws the hand landmarks and connections using MediaPipe's drawing utilities.
            - Optionally calls `draw_landmark_names` depending on `self.capture_mode`.
            - Calls `recognize_gesture` to try to match the hand against database gestures.
        """
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label 
            score = handedness.classification[0].score

            if label == "Right":
                hand_color = (235, 137, 52)
            else:
                hand_color = (235, 52, 113)

            score_color = (0, 255, int(255 * (1 - score))) 

            if draw:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=hand_color, thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=hand_color, thickness=2)
                )

            self._draw_landmark_names(image, hand_landmarks)
            bbox = self._bounding_box(image, hand_landmarks.landmark)

            
    def _draw_landmark_names(self, image, hand_landmarks):
        """
        Draw landmark indices or coordinates next to each hand landmark

        Args:
            image (numpy.ndarray):
                Current frame (BGR) where the text labels will be drawn.
            hand_landmarks:
                A MediaPipe `NormalizedLandmarkList` instance for a SINGLE HAND
                (the object from `results.multi_hand_landmarks[i]`).
                THIS IS THE HAND WITH THE LANDMARKS WITHIN THE ARRAY.
            mode (str):
                Controls what text is displayed:
                    - "landmarks"         → only the landmark index (0-20).
                    - "landmarks_coords"  → index + normalized (x, y) coordinates.
        """
        if self.capture_mode not in ["landmarks", "landmarks_coords"]:
            return
        for i, landmark in enumerate(hand_landmarks.landmark): # Iterate over each landmark of the hand
                # Depending on the capture mode, display differents texts
                coords = ""
                if self.capture_mode == "landmarks_coords":
                    coords = f"{i}: ({landmark.x:.2f}, {landmark.y:.2f})"
                elif self.capture_mode == "landmarks":
                    coords = f"{i}"
                    
                width, height = self._get_frame_dimensions(image)
                px = int(landmark.x * width)
                py = int(landmark.y * height)
                cv.putText(img=image, text=coords, org=(px + 30, py), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=(0, 0, 0), lineType = cv.LINE_AA)
            
    def _bounding_box(self, image, hand_landmarks) -> tuple:
        """
        Draw a bounding box around the hand and return its coordinates
        The bounding box is computed using the min/max of the normalized
        x/y coordinates of all landmarks, then expanded with a fixed padding.

        Args:
            image (numpy.ndarray):
                Current frame (BGR) where the rectangle will be drawn.
            hand_landmarks (Sequence[NormalizedLandmark]):
                Iterable of 21 MediaPipe landmarks (`hand_landmarks.landmark`).
                THIS THE HAND, THE ARRAY OF LANDMARKS.

        Returns:
            tuple[int, int, int, int]:
                (x1, y1, x2, y2) pixel coordinates of the bounding box:
                - (x1, y1) → top-left corner
                - (x2, y2) → bottom-right corner
        """
        if self.capture_mode == "bounding_box":
            width, height = self._get_frame_dimensions(image)
            xs = [landmark.x for landmark in hand_landmarks]
            ys = [landmark.y for landmark in hand_landmarks]

            if not xs or not ys:
                return None

            x_min = min(xs)
            x_max = max(xs)
            y_min = min(ys)
            y_max = max(ys)

            padding = 0.02
            x_min = max(0.0, x_min - padding)
            y_min = max(0.0, y_min - padding)
            x_max = min(1.0, x_max + padding)
            y_max = min(1.0, y_max + padding)

            x1 = int(x_min * width)
            y1 = int(y_min * height)
            x2 = int(x_max * width)
            y2 = int(y_max * height)

            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            return (x1, y1, x2, y2)
        else:
            return None

    
    def _get_frame_dimensions(self, image):
        """
        Returns (width, height) for both videocapture frames and static images.

        - If using webcam, any other numerical device or video (self.cap exists), 
          uses cap.get() because the frame might be resized by the camera backend.

        - If using static image, falls back to image.shape.
        """
        if hasattr(self, "cap") and self.cap is not None:
            w = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            if w > 0 and h > 0:
                return w, h

        # Fallback for images (numpy array)
        h, w = image.shape[:2]
        return w, h
    
class Hand():
    
    def __init__(self, side: str, gesture, landmarks: list):
        self.side = side
        self.sound_file_path = None
        self.landmarks = landmarks
        # gesture is a Category from MediaPipe
        self.gesture = gesture.category_name
        self.landmarks_origin = (
            min(land.x for land in self.landmarks),
            min(land.y for land in self.landmarks)
        )
        self.landmarks_end = (
            max(land.x for land in self.landmarks),
            max(land.y for land in self.landmarks)
        )
    
    def getSoundFilePath(self):
        return self.sound_file_path
    
class DJ():
    
    _AUDIO_MOKE_FILE = "C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose/assets/boing.mp3"
    _BASE = "C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose"
    _SF2=r"assets\FluidR3_GM.sf2"
    
    def __init__(self):
        pygame.mixer.init()
        self.ch = pygame.mixer.Channel(0)
        
        # FluidS
        self._fs = fluidsynth.Synth()
        self._fs.start(driver="dsound")
        self._sfid = self._fs.sfload(self._SF2)
        
        # Logic to work beyond the frames loops and with rests
        self.cooldown_s = 0.74
        self._last_combo = None
        self._last_t = 0.0
        
        self.programs = {
            "Open_Palm": 0, # Pian
            "ILoveYou": 33, # Bass
            "Victory": 29, # Eletric Guitar
            "Pointing_Up": 80, # Synth
            "Thumb_Up": 45 # Low Tom
        }
        
        self.notes = {
            "Open_Palm": 48, # C3 Dó
            "ILoveYou": 62, # D4 Ré
            "Victory": 69, # A4 Lá
            "Pointing_Up": 64, # E4 Mi
            "Thumb_Up": 65 # F4 Fá
        }
        
    def _play_note(self, ch, prog, note, vel=127, dur=0.74, bank=0):
        self._fs.program_select(ch, self._sfid, bank, prog)
        self._fs.noteon(ch, note, vel)
        Timer(dur, lambda: self._fs.noteoff(ch, note)).start() # Making a queue of notes
            
    def play_sound(self, right_hand, left_hand):
        valid = ["Open_Palm", "ILoveYou", "Victory", "Pointing_Up", "Thumb_Up"]
        # rest, to allow the same sound or just a semibreve rest 
        if right_hand.gesture == "Closed_Fist" or left_hand.gesture == "Closed_Fist":
            self._last_combo = None
            return
        
        if right_hand.gesture not in valid or left_hand.gesture not in valid:
            self._last_combo = None
            return
        
        program = self.programs[left_hand.gesture]
        note = self.notes[right_hand.gesture]
        
        # No reapet logic
        combo = (program, note)
        now = time.time()
        if combo == self._last_combo:
            return
        if now - self._last_t < self.cooldown_s:
            return
        if self.ch.get_busy():
            return
        self._last_combo = combo
        self._last_t = now
        
        self._play_note(0, prog=program, note=note)
        
class HandSpeller():
    """
    The Gesture Recognizer
    """
    _MODEL_PATH = "C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose/gesture_recognizer.task"
    
    def __init__(self, model=_MODEL_PATH, running_mode=vision.RunningMode.IMAGE):
        self.base_options = python.BaseOptions(model_asset_path=model)
        self.options = vision.GestureRecognizerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.IMAGE,#running_mode,
            num_hands=2
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)
        self.dj = DJ()
    
    def process_image(self, image, w, h):
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        results = self.recognizer.recognize(mp_image)
        if not results.gestures or not results.hand_landmarks:
            return image, None

        hands_by_side = {}

        for i, landmarks in enumerate(results.hand_landmarks):
            gesture_category = results.gestures[i][0]
            side = results.handedness[i][0].category_name  # assets to be right and left

            hand = Hand(side=side, gesture=gesture_category, landmarks=landmarks)
            hands_by_side[side] = hand

            x = int((hand.landmarks_origin[0] * w) - 30)
            y = int((hand.landmarks_origin[1] * h) - 30)
            cv.putText(
                img=image,
                text=hand.gesture,
                org=(x, y),
                fontFace=cv.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                thickness=1,
                color=(120, 23, 190),
                lineType=cv.LINE_AA
            )

        if "Right" in hands_by_side and "Left" in hands_by_side:
            self.dj.play_sound(hands_by_side["Left"], hands_by_side["Right"]) # Inverted because we flip at the opencv

        return image, results

