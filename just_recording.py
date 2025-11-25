import cv2 as cv
import mediapipe as mp
import pygame
import json


class Recorder():
    def __init__(self, name="Just Compose Beta", device=0, capture_mode=None):
        """
        Initialize the Recorder instance

        Args:
            name (str, optional):
                Window title used by OpenCV when displaying the frames.
                Defaults to "Just Compose Beta".
            device (int | str, optional):
                Capture source.
                - If `int`, it is treated as an OpenCV camera index (0 for default webcam).
                - If `str` and the path ends with a compatible url
                Defaults to 0.
            capture_mode (str | None, optional):
                Controls which text overlays are drawn on top of the detected landmarks.
                Supported values:
                    - "landmarks"         → draw only the landmark indices
                    - "landmarks_coords"  → draw landmark indices + normalized coordinates
                    - None                → do not draw any landmark labels
                Defaults to None.
        """
        # MediaPipe structure
        self.mp_hands = mp.solutions.hands 
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Class attributes
        self.name = name
        self.device = device
        self.capture_mode = capture_mode
        self.pose_cache = []
        self.logical_pose_cache = []
    
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
                    if key == ord("s") and results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):    
                            gesture = self.snapshot_hand_pose(hand_landmarks=hand_landmarks.landmark, handedness_label=handedness.classification[0].label)
                            #print("Snapshot taken:")
                            #print(json.dumps(gesture, indent=2))
                    
                    if cv.getWindowProperty(self.name, cv.WND_PROP_VISIBLE) < 1:
                        break
                    
                # Break of the loop -> Release resources
                self.cap.release()
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
            self.draw_bounding_box(image, hand_landmarks.landmark)
    
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
                    
                width  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
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
        width  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
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
    
    def snapshot_hand_pose(self, hand_landmarks: list, handedness_label: str) -> dict:
        """
        Capture the current hand pose (normalized landmarks) as a template dict
        
        Args:
            hand_landmarks (Sequence[NormalizedLandmark]):
                Iterable of 21 MediaPipe landmarks (`hand_landmarks.landmark`).
                THIS THE HAND, THE ARRAY OF LANDMARKS.
            handedness_label (str):
                "Left" or "Right" label for the detected hand.
        """
        pose = {
            "hand_side": handedness_label.lower(),  # "left" / "right"
            "landmarks": []
        }

        for i, lm in enumerate(hand_landmarks):
            pose["landmarks"].append({
                "id": i,
                "x": lm.x,
                "y": lm.y,
            })

        self.pose_cache.append(pose)
        self.pose_logical_representation(pose)
        return pose
    
    def pose_logical_representation(self, pose):
        print("Computing logical representation for the captured pose...")
        index_finger = (pose["landmarks"][6:9])
        print(index_finger)
        index_finger_y = (pose["landmarks"][8]["y"], pose["landmarks"][6]["y"])
        pink_finger_y = (pose["landmarks"][20]["y"], pose["landmarks"][18]["y"])
        midfinger_y = (pose["landmarks"][12]["y"], pose["landmarks"][10]["y"])
        thumb_x = (pose["landmarks"][4]["x"], pose["landmarks"][2]["x"])
        logical_pose = {
            "hand_side": pose["hand_side"],
            "index_finger": "up" if index_finger_y[0] < index_finger_y[1] else "down",
            "middle_finger": "up" if midfinger_y[0] < midfinger_y[1] else "down",
            "pink_finger": "up" if pink_finger_y[0] < pink_finger_y[1] else "down",
            "thumb": "open" if thumb_x[0] > thumb_x[1] else "closed"
        }
        print("Logical representation:", logical_pose)
    
if __name__ == "__main__":
    recorder = Recorder(name="Just Compose Beta", device=0, capture_mode="landmarks_coords")
    recorder.capture()