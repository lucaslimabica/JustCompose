# Responsible Camera for Motion Capture
import cv2 as cv
import mediapipe as mp
import pygame
import database_manager


class Camera:
    """
    Main camera handler for motion capture and gesture recognition.
    This class is responsible for:
    - Capturing frames from a video source (webcam or image file)
    - Running MediaPipe Hands on each frame
    - Drawing landmarks, labels, and bounding boxes
    - Recognizing gestures based on conditions stored in the database
    """
    
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
                    - None                → do not draw any landmark labels
                Defaults to None.
        """
        # Pygame structure
        self.mixer = pygame.mixer
        self.mixer.init()
        self.audio_channel = self.mixer.Channel(0)
        self.boing = self.mixer.Sound("C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose/assets/boing.mp3")

        # MediaPipe structure
        self.mp_hands = mp.solutions.hands 
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Class attributes
        self.name = name
        self.device = device
        self.compatible_file_types = ('.jpg', '.jpeg', '.png')
        self.capture_mode = capture_mode 
    
    def capture(self):
        """
        Start the capture loop AND process frames from the configured device with MediaPipe Hands

        Behavior:
            - If `self.device` is an integer:
                * Opens a VideoCapture stream.
                * Processes frames in real time.
                * Runs MediaPipe Hands on each frame.
                * Draws landmarks and recognized gestures if any.
            - If `self.device` is a string and points to an image file:
                * Loads the image.
                * Runs MediaPipe Hands once.
                * Draws landmarks and recognized gestures.

        This method blocks until:
            - The window is closed, or
            - The user presses ESC, 'q', or 'l'.
        """
        if self.device == 0:        
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
                    # Make detections
                    results = hand_detector.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks:  # Avoid None for the drawing func
                        self.draw_landmarks(frame, results)

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

                
        elif isinstance(self.device, str) and self.device.endswith(self.compatible_file_types):
            image = cv.imread(self.device)

            with self.mp_hands.Hands(static_image_mode=True) as hand_detector:
                results = hand_detector.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    self.draw_landmarks(image, results)

                cv.imshow(self.name, image)

                while True:
                    key = cv.waitKey(1) 
                    # Keyboard LEGACY
                    if key in [27, ord("q"), ord("l")]:
                        break
                    if cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) < 1:
                        break
                    
                cv.destroyAllWindows()

                
    def draw_landmarks(self, image, results):
        """
        Draw hand landmarks, connections, labels, and recognized gestures
        Args:
            image (numpy.ndarray):
                Current frame (BGR) where the landmarks and overlays will be drawn.
            results (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
                The result object returned from `mp_hands.Hands.process(...)`,
                containing `multi_hand_landmarks` and `multi_handedness`.

        Behavior:
            - Iterates over each detected hand and its handedness.
            - Draws the hand landmarks and connections using MediaPipe's drawing utilities.
            - Optionally calls `draw_landmark_names` depending on `self.capture_mode`.
            - Calls `recognize_gesture` to try to match the hand against database gestures.
        """
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label # classification is a list of all possible classes for the hand, so the 0 is the more accurate one
            score = handedness.classification[0].score
            if label == "Right":
                hand_color = (235, 137, 52)
            else:
                hand_color = (235, 52, 113)
            score_color = (0, 255, int(255 * (1 - score))) # from yellow to green based on the score
            
            # Then, draw the landmarks on the frame
            self.mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=score_color, thickness=1, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=hand_color, thickness=1, circle_radius=1)
            )
            # 1º arg: image to draw on
            # 2º arg: landmarks to draw
            # 3º arg: the connections between the landmarks
            # 4º arg: style for the circles (landmarks)
            # ==========================================================
            if self.capture_mode in ["landmarks", "landmarks_coords"]:
                self.draw_landmark_names(image, hand_landmarks, self.capture_mode)
            self.recognize_gesture(image=image, hand_landmarks=hand_landmarks.landmark, label=label)
            
    def draw_landmark_names(self, image, hand_landmarks, mode):
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
                    - "landmarks"         → only the landmark index (0–20).
                    - "landmarks_coords"  → index + normalized (x, y) coordinates.
        """
        for i, landmark in enumerate(hand_landmarks.landmark): # Iterate over each landmark of the hand
                # Depending on the capture mode, display differents texts
                coords = ""
                if mode == "landmarks_coords":
                    coords = f"{i}: ({landmark.x:.2f}, {landmark.y:.2f})"
                elif mode == "landmarks":
                    coords = f"{i}"
                    
                width, height = self.get_frame_dimensions(image)
                px = int(landmark.x * width)
                py = int(landmark.y * height)
                cv.putText(img=image, text=coords, org=(px + 30, py), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=(0, 0, 0), lineType = cv.LINE_AA)
            
    def draw_bounding_box(self, image, hand_landmarks) -> tuple:
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
        width, height = self.get_frame_dimensions(image)
        xs = [landmark.x for landmark in hand_landmarks]
        ys = [landmark.y for landmark in hand_landmarks]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        x1 = int(min_x * width) - 20
        y1 = int(min_y * height) - 20
        x2 = int(max_x * width) + 20
        y2 = int(max_y * height) + 20
        cv.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)
        return (x1, y1, x2, y2)
        
    def recognize_gesture(self, image, hand_landmarks: list, label: str):
        """
        Recognize a gesture for a single hand and draw its name on the image, abocve the bounding box

        Args:
            image (numpy.ndarray):
                Current frame (BGR) where the gesture name will be drawn if recognized.
            hand_landmarks (list):
                List-like container of MediaPipe `NormalizedLandmark` objects
                (`hand_landmarks.landmark` from MediaPipe).
                THIS IS THE HAND, THE ARRAY OF LANDMARKS.
            label (str):
                Handedness label returned by MediaPipe, usually `"Right"` or `"Left"`.

        Behavior:
            - Draws a bounding box around the hand.
            - Loads all gestures and conditions from the database.
            - Attempts to match the current hand landmarks against each gesture.
            - If a gesture matches, its name is rendered above the bounding box.
        """
        hand = self.draw_bounding_box(image, hand_landmarks)
        gestures = database_manager.load_all_gestures() # load gestures from the database as dicts
        detected = self.recognize_gesture_from_db(hand_landmarks, label, gestures)
        if detected:
            cv.putText(image, detected["name"], org=(hand[0], hand[1]-10), fontFace=cv.FONT_HERSHEY_SIMPLEX,fontScale= 1, color=(0,255,0), thickness=2)
            
    def condition_is_true(self, hand_landmarks, handedness_label, cond) -> bool:
        """
        Evaluate a single gesture condition against the current hand landmarks

        Args:
            hand_landmarks (list):
                List-like container of MediaPipe `NormalizedLandmark` objects
                REPRESENTING A SINGLE HAND
            handedness_label (str):
                Handedness label, `"Right"` or `"Left"`.
            cond (dict):
                Condition dictionary loaded from the database, with keys:
                    - "a":    index of the first landmark (int)
                    - "b":    index of the second landmark (int)
                    - "op":   comparison operator (str: "<", ">", "<=", ">=", "==")
                    - "axis": coordinate axis to compare ("x", "y", or "z")
                    - "side": which hand this condition applies to ("left", "right", "any")

        Returns:
            bool:
                True if the condition is satisfied (or does not apply to this hand side),
                False otherwise.
        """
        if cond["side"] != "any" and cond["side"] != handedness_label.lower():
            return True  # Condition does not apply to this hand

        la = hand_landmarks[cond["a"]]
        lb = hand_landmarks[cond["b"]]

        va = getattr(la, cond["axis"])
        vb = getattr(lb, cond["axis"])

        op = cond["op"]

        if op == "<":  return va < vb
        if op == ">":  return va > vb

        return False

    def recognize_gesture_from_db(self, hand_landmarks, handedness_label, gestures_db):
        """
        Match the current hand landmarks against all gestures in the database.

        Args:
            hand_landmarks (list):
                List-like container of MediaPipe `NormalizedLandmark` objects
                REPRESENTING A SINGLE HAND
            handedness_label (str):
                Handedness label, `"Right"` or `"Left"`.
            gestures_db (dict):
                Dictionary of gestures as loaded from the database. Expected format:
                {
                    gesture_id: {
                        "name": str,
                        "description": str,
                        "sound": str,
                        "conditions": [
                            {"a": int, "op": str, "b": int, "axis": str, "side": str},
                            ...
                        ]
                    },
                    ...
                }

        Returns:
            dict | None:
                The first matching gesture dictionary if all its conditions
                evaluate to True, or None if no gesture matches.
        """
        for gid, gesture in gestures_db.items():
            match = True

            for cond in gesture["conditions"]:
                if not self.condition_is_true(hand_landmarks, handedness_label, cond):
                    match = False # condition failed cause its different
                    break

            if match:
                return gesture  # find the first matching gesture

        return None
    
    def get_frame_dimensions(self, image):
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