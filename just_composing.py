# Responsible Camera for Motion Capture
import cv2 as cv
import mediapipe as mp
import pygame
import database_manager
import math


# TODO: TÁ MUITO SENSÍVEL A CAPTURA, talvez diminuir as conds do move

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
        Evaluate a single gesture condition against the current hand landmarks.

        This function acts as a unified dispatcher for all supported condition types:
        - **"bin"**:     Axis-wise comparison between two landmarks (A.axis < B.axis, etc.)
        - **"delta"**:   Signed difference test (A.axis - B.axis < threshold)
        - **"distance"**: Euclidean distance test between landmarks A and B,
                          optionally normalized (e.g., by hand width).

        Before evaluating the condition, the function also checks if the rule
        applies to the detected hand side ("left", "right", or "any").

        ----------------------------------------------------------------------
        Parameters
        ----------------------------------------------------------------------
        hand_landmarks : list[NormalizedLandmark]
            A list (indexable 0-20) of MediaPipe `NormalizedLandmark` objects
            representing **one single detected hand**.

        handedness_label : str
            Detected handedness of this hand ("Left" or "Right").
            Used to filter out conditions that apply only to one side.

        cond : dict
            Condition record loaded from the gesture database. Expected keys:

            Required:
                - "a"   (int): landmark index A
                - "b"   (int): landmark index B
                - "op"  (str): comparison operator ("<", ">", "<=", ">=", "==")
                - "side" (str): "left", "right", or "any"

           Depending on type:
                - type "bin":
                    - "axis" (str): "x", "y", or "z"
                    Compares A.axis with B.axis using operator.

                - type "delta":
                    - "axis" (str): "x", "y", or "z"
                    - "threshold" (float): threshold for (A.axis - B.axis)
                    Evaluates: (A.axis - B.axis) <op> threshold

                - type "distance":
                    - "threshold" (float): threshold to compare against distance(A,B)
                    - "normalize_by" (str | None):
                          One of: None, "hand_width", "hand_height".
                    Evaluates: distance(A,B) <op> threshold*(normalization)

            Optional:
                - "type" (str):
                      One of: "bin" (default), "delta", "distance"
                - "weight" (float):
                      Condition weight (used by the gesture classifier).

        ----------------------------------------------------------------------
        Returns
        ----------------------------------------------------------------------
        bool
            True if the condition is satisfied; False otherwise.

            Conditions that do **not apply** to the current hand side
            (cond["side"] == "left" and detected hand is right)
            are treated as **automatically satisfied** enabling
            symmetrical gestures, if we had one.
        """
        side = cond.get("side", "any").lower()
        hand = handedness_label.lower()
        if side != "any" and side != hand:
            return True
        
        ctype = cond.get("type", "bin")

        if ctype == "bin":
            return self._cmp_condition(hand_landmarks, cond)
        elif ctype == "delta":
            return self._delta_condition(hand_landmarks, cond)
        elif ctype == "distance":
            return self._distance_condition(hand_landmarks, cond)

        return False

    
    def _bin_condition(self, hand_landmarks, cond, tol=0.02):
        """
        Evaluate a binary axis-wise comparison between two landmarks.

        This is the base comparison used for "bin" (binary/boolean) conditions,
        corresponding to inequalities such as:

            A.axis < B.axis
            A.axis > B.axis
            A.axis == B.axis
            ...

        A small tolerance (`tol`) is applied to reduce noise from
        MediaPipe's subpixel landmark jitter.

        ----------------------------------------------------------------------
        Parameters
        ----------------------------------------------------------------------
        hand_landmarks : list[NormalizedLandmark]
            List of 21 MediaPipe hand landmarks for a single hand.

        cond : dict
            Condition with keys:
                - "a"     (int): landmark index A
                - "b"     (int): landmark index B
                - "axis"  (str): "x", "y", or "z"
                - "op"    (str): "<", ">", "<=", ">=", "=="

        tol : float
            Noise compensation tolerance added to the comparison.
            For example, "<" becomes `(va < vb + tol)`.

        ----------------------------------------------------------------------
        Returns
        ----------------------------------------------------------------------
        bool
            True if comparison (with tolerance) holds.
        """
        la = hand_landmarks[cond["a"]]
        lb = hand_landmarks[cond["b"]]
        va = getattr(la, cond["axis"])
        vb = getattr(lb, cond["axis"])
        op = cond["op"]

        if op == "<": return va < vb + tol
        if op == ">": return va > vb - tol
        if op == "<=": return va <= vb + tol
        if op == ">=": return va >= vb - tol
        if op == "==": return abs(va - vb) < tol
        
        return False
    
    def _delta_condition(self, hand_landmarks, cond):
        """
        Evaluate a signed difference condition between two landmarks
        along a chosen axis.

        This checks conditions of the form:

            (A.axis - B.axis) < threshold
            (A.axis - B.axis) > threshold

        Delta-based conditions measure *how much* one landmark is above,
        below, left, or right of another, instead of only the direction.
        This is useful for "finger clearly up", "thumb clearly open",
        or any logic requiring a magnitude in the comparison.

        ----------------------------------------------------------------------
        Parameters
        ----------------------------------------------------------------------
        hand_landmarks : list[NormalizedLandmark]
            List of 21 hand landmarks for the current detected hand.

        cond : dict
            Condition with keys:
                - "a"         (int): landmark A
                - "b"         (int): landmark B
                - "axis"      (str): "x", "y", or "z"
                - "threshold" (float): threshold for the delta
                - "op"        (str): "<" or ">"

        ----------------------------------------------------------------------
        Returns
        ----------------------------------------------------------------------
        bool
            True if the signed delta comparison is satisfied.
        """
        la = hand_landmarks[cond["a"]]
        lb = hand_landmarks[cond["b"]]
        va = getattr(la, cond["axis"])
        vb = getattr(lb, cond["axis"])

        delta = va - vb
        thr = cond.get("threshold", 0.0)
        op = cond["op"]

        if op == "<": return delta < thr
        if op == ">": return delta > thr
        return False

    def _distance_condition(self, hand_landmarks, cond):
        """
        Evaluate a distance-based condition between two landmarks.

        Computes the Euclidean distance:

            dist = sqrt( (Ax - Bx)^2 + (Ay - By)^2 + (Az - Bz)^2 )

        Optionally normalizes the distance by:
            - "hand_width":  max(x) - min(x)

        This allows gesture rules to remain stable regardless of:
            * hand size
            * camera distance
            * perspective distortion

        Distance-based rules are ideal for:
            - pinch gestures (thumb-index proximity)
            - V shapes (index-middle separation)
            - open-hand vs closed-fist measurements
            - zoom-like gestures (distance increasing/decreasing)

        ----------------------------------------------------------------------
        Parameters
        ----------------------------------------------------------------------
        hand_landmarks : list[NormalizedLandmark]
            List of 21 MediaPipe hand landmarks (single hand).

        cond : dict
            Condition with keys:
                - "a"             (int): landmark A
                - "b"             (int): landmark B
                - "op"            (str): "<" or ">"
                - "threshold"     (float): distance threshold
                - "normalize_by"  (str | None):
                        None
                        "hand_width"
                        "hand_height"   (future)
                        "palm_size"     (future)

        ----------------------------------------------------------------------
        Returns
        ----------------------------------------------------------------------
        bool
            True if the (normalized) distance satisfies the comparison.
        """
        la = hand_landmarks[cond["a"]]
        lb = hand_landmarks[cond["b"]]

        dx = la.x - lb.x
        dy = la.y - lb.y
        dz = la.z - lb.z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        norm = cond.get("normalize_by", None)
        if norm == "hand_width":
            xs = [lm.x for lm in hand_landmarks]
            hand_width = max(xs) - min(xs)
            dist /= hand_width or 1.0

        thr = cond.get("threshold", 0.0)
        op = cond["op"]

        if op == "<": return dist < thr
        if op == ">": return dist > thr

        return False

    def recognize_gesture_from_db(self, hand_landmarks, handedness_label, gestures_db, min_score=0.7):
        """
        Match the current hand landmarks against all gestures in the database,
        using a weighted score over logical conditions.

        Each gesture is defined as a set of conditions (boolean rules). For every
        gesture, this method computes:

            score = (sum of weights of satisfied conditions) / (sum of all weights)

        The gesture with the highest score above `min_score` is returned.

        ----------------------------------------------------------------------
        Parameters
        ----------------------------------------------------------------------
        hand_landmarks : list[NormalizedLandmark]
            List-like container of MediaPipe `NormalizedLandmark` objects
            representing a **single hand** (21 landmarks, indices 0–20).

        handedness_label : str
            Detected handedness for this hand: `"Right"` or `"Left"`.
            Used by `condition_is_true` to ignore conditions that apply
            only to the opposite side (e.g. rules for `"left"` when
            evaluating a `"Right"` hand).

        gestures_db : dict[int, dict]
            Dictionary of gestures as loaded from the database. Expected format:

            {
                gesture_id: {
                    "name": str,
                    "description": str,
                    "sound": str,
                    "conditions": [
                        {
                            "type": "bin" | "delta" | "distance",
                            "a": int,
                            "b": int,
                            "op": str,
                            "axis": str | None,
                            "side": "left" | "right" | "any",
                            "threshold": float | None,
                            "normalize_by": str | None,
                            "weight": float   # optional, defaults to 1.0
                        },
                        ...
                    ]
                },
                ...
            }

        min_score : float, optional
            Minimum normalized score required to accept a gesture.
            Range is [0.0, 1.0]. For example:
                - 0.7 → tolerant (some conditions may fail)
                - 0.9 → strict (almost all conditions must pass)

        ----------------------------------------------------------------------
        Returns
        ----------------------------------------------------------------------
        dict | None
            The gesture dictionary with the **highest score** greater than
            or equal to `min_score`, or `None` if no gesture reaches the
            required score.

            Example returned gesture:

            {
                "name": "Number One",
                "description": "...",
                "sound": "./assets/boing.mp3",
                "conditions": [...]
            }
        """
        best_gesture = None
        best_score = 0

        for gid, gesture in gestures_db.items():
            conds = gesture["conditions"]

            total_weight = sum(c.get("weight", 1.0) for c in conds)
            if total_weight == 0:
                continue

            passed_weight = 0

            for cond in conds:
                if self.condition_is_true(hand_landmarks, handedness_label, cond):
                    passed_weight += cond.get("weight", 1.0)

            score = passed_weight / total_weight

            # Debug (optional)
            # print(f"[DEBUG] {gesture['name']} score: {score:.2f}")

            # Track best gesture
            if score > best_score:
                best_score = score
                best_gesture = gesture

        # Only return gesture if score is high enough
        if best_gesture and best_score >= min_score:
            return best_gesture

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