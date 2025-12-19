# Responsible Camera for Motion Capture
import cv2 as cv
import mediapipe as mp
import pygame
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
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
    """
    Represents a detected hand and its current state.
    
    Attributes:
        side (str): "Left" or "Right".
        gesture (str): The name of the recognized gesture (e.g., "Open_Palm")
        landmarks (list): List of MediaPipe landmark objects
        landmarks_origin (tuple): (x, y) normalized coordinates of the top-left most point
        landmarks_end (tuple): (x, y) normalized coordinates of the bottom-right most point
    """
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
    """
    Audio controller responsible for synthesizing sounds based on hand gestures.
    Uses FluidSynth to generate MIDI events.
    
    This class manages:
        - Loading SoundFonts (.sf2)
        - Mapping gesture names to MIDI program IDs (instruments)
        - Mapping gesture names to MIDI notes
        - Debouncing sound triggering (cooldowns)
    """
    _AUDIO_MOKE_FILE = "C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose/assets/boing.mp3"
    _BASE = "C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose"
    _SF2=r"assets\FluidR3_GM.sf2"
    
    def __init__(self):
        """Initialize the synthesizer and load the sound bank from .sf2"""
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
        """
        Internal helper to trigger a MIDI note on/off event sequence.
        
        Args:
            ch (int): MIDI channel
            prog (int): Program number (Instrument ID)
            note (int): MIDI note number
            vel (int): Velocity (volume/intensity), 0-127
            dur (float): Duration in seconds before sending note-off.
            bank (int): Sound bank ID
        """
        self._fs.program_select(ch, self._sfid, bank, prog)
        self._fs.noteon(ch, note, vel)
        Timer(dur, lambda: self._fs.noteoff(ch, note)).start() # Making a queue of notes
            
    def play_sound(self, right_hand, left_hand):
        """
        Determines if a sound should be played based on the current gestures of both hands.
        
        Logic:
            - Left Hand determines the Instrument (Program)
            - Right Hand determines the Note
            - Prevents spamming by checking a cooldown timer
            - Ignores invalid gestures or "Closed_Fist" (Rest)
        
        Args:
            right_hand (Hand): The detected right hand object.
            left_hand (Hand): The detected left hand object.
        """
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
    The Gesture Recognizer and UI manager
    Handles the interaction between the visual recognition (MediaPipe) and the Logic/Audio (DJ)
    """
    _MODEL_PATH = "C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose/gesture_recognizer.task"
    
    def __init__(self, model=_MODEL_PATH, running_mode=vision.RunningMode.IMAGE):
        """
        Initializes the HandSpeller.

        Args:
            model (str): Path to the .task file for gesture recognition.
            running_mode: MediaPipe running mode (IMAGE, VIDEO, or LIVE_STREAM)
        """
        self.base_options = python.BaseOptions(model_asset_path=model)
        self.options = vision.GestureRecognizerOptions(
            base_options=self.base_options,
            running_mode=vision.RunningMode.IMAGE,#running_mode,
            num_hands=2
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(self.options)
        self.dj = DJ()
        
        # Icons images
        self.electric_guitar = cv.imread("./assets/electric-guitar.png", cv.IMREAD_UNCHANGED)
        self.bass = cv.imread("./assets/bass.png", cv.IMREAD_UNCHANGED)
        self.synth = cv.imread("./assets/synth.png", cv.IMREAD_UNCHANGED)
        self.piano = cv.imread("./assets/piano.png", cv.IMREAD_UNCHANGED)
        self.tom = cv.imread("./assets/drum.png", cv.IMREAD_UNCHANGED)
        self.rest = cv.imread("./assets/rest.png", cv.IMREAD_UNCHANGED)
        self.icons = {
            "Piano": self.piano,
            "Bass": self.bass,
            "Eletric Guitar": self.electric_guitar,
            "Synth": self.synth,
            "Low Tom": self.tom,
            "Rest": self.rest,
        }

    def _overlay_png(self, frame, png, x=20, y=20, size=64):
        """
        Overlays a PNG image with transparency (alpha channel) onto the background frame.
        
        Args:
            frame (numpy.ndarray): The background image (BGR). Modified in-place
            png (numpy.ndarray): The overlay image (must include Alpha channel, BGRA)
            x (int): X-coordinate for top-left corner
            y (int): Y-coordinate for top-left corner
            size (int): Target width/height to resize the icon to (square aspect)
        """
        if png is None:
            return
        if png.shape[2] != 4:
            return

        # resize
        png_resized = cv.resize(png, (size, size), interpolation=cv.INTER_AREA)
        h, w = png_resized.shape[:2]

        # asserting the size inside the frame
        H, W = frame.shape[:2]
        if x + w > W or y + h > H:
            return

        alpha = png_resized[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = (alpha * png_resized[:, :, c] + (1 - alpha) * frame[y:y+h, x:x+w, c])

    
    def _gesture_to_icon(self, gesture, label):
        """
        Maps a raw MediaPipe gesture to a display name or instrument string

        Args:
            gesture: The gesture object from MediaPipe results
            label (str): "Right" or "Left" indicating the hand side

        Returns:
            str: Name of the instrument (if Right hand) or Note name (if Left hand)
        """
        name = getattr(gesture, "category_name", None)
        if not name or name == "None":
            return ""
        
        instrument = {
            "Open_Palm": "Piano",
            "ILoveYou": "Bass",
            "Victory": "Eletric Guitar",
            "Pointing_Up": "Synth",
            "Thumb_Up": "Low Tom",
            "Closed_Fist": "Rest"
        }
        
        notes = {
            "Open_Palm": "C3 / DO",
            "ILoveYou": "D4 / RE",
            "Victory": "A4 / LA",
            "Pointing_Up": "E4 / MI",
            "Thumb_Up": "F4 / FA",
            "Closed_Fist": "Rest"
        }
        
        try:
            if label == "Right":
                return instrument[gesture.category_name]
            else:
                return notes[gesture.category_name]
        except KeyError:
            return "Rest"
    
    def process_image(self, image, w, h):
        """
        Core processing loop for the HandSpeller logic.
        
        1. Recognizes gestures in the image
        2. Draws the instrument icons and text overlays
        3. Sends the recognized hand states to the DJ for audio playback

        Args:
            image (numpy.ndarray): Input frame (BGR)
            w (int): Width of the frame
            h (int): Height of the frame

        Returns:
            tuple: (processed_image, recognition_results)
        """
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
            self.last_hands_by_side = hands_by_side

            instrument = self._gesture_to_icon(gesture_category, side)
            
            x = int((hand.landmarks_origin[0] * w) - 30)
            y = int((hand.landmarks_origin[1] * h) - 30)
            if side == "Left":
                cv.putText(
                    img=image,
                    text=instrument,
                    org=(x, y),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness=1,
                    color=(0, 0, 0),
                    lineType=cv.LINE_AA
                )
            try:
                icon = self.icons.get(instrument)
                self._overlay_png(image, icon, x=20, y=20, size=80)
            except:
                pass
            
        if "Right" in hands_by_side and "Left" in hands_by_side:
            self.dj.play_sound(hands_by_side["Left"], hands_by_side["Right"]) # Inverted because we flip at the opencv
        
        return image, results

