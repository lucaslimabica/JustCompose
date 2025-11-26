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
                            print("Snapshot taken:")
                            print(json.dumps(gesture, indent=2))
                    
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
        pose = self.pose_logical_representation(pose)
        return pose
    
    def pose_logical_representation(self, pose):
        """
        Convert a captured pose (raw landmarks) into a logical representation.

        It computes:
          - which fingers are up/down (Y axis)
          - thumb open/closed (X axis)
          - a set of logical conditions (tuples) that can be stored as gesture rules

        Args:
            pose (dict):
                {
                  "hand_side": "left" | "right",
                  "landmarks": [
                    {"id": 0, "x": ..., "y": ..., "z": ...},
                    {"id": 20, "x": ..., "y": ..., "z": ...},
                  ]
                }
        Returns:
            dict:
                Logical representation of the pose:
                {
                  "name": "Number Three",
                  "hand_side": "right",
                  "index_finger": "up",
                  "middle_finger": "up",
                  "ring_finger": "down",
                  "pink_finger": "down",
                  "thumb": "open",
                  "conds": [
                    (6, ">", 8, "y", "right"),
                    (10, ">", 12, "y", "right"),
                    (14, "<", 16, "y", "right"),
                    (18, ">", 20, "y", "right"),
                    (2, "<", 4, "x", "right"),
                    (8, ">", 2, "x", "right"),
                    (8, "<", 10, "x", "right"),
                    (12, "<", 6, "x", "right"),
                    (12, "<", 14, "x", "right"),
                    (16, ">", 10, "x", "right"),
                    (16, ">", 18, "x", "right"),
                    (20, ">", 14, "x", "right")
                  ]
                }
        """
        print("Computing logical representation for the captured pose...")

        lms = pose["landmarks"]
        side = pose["hand_side"]   # "left" or "right"

        # --- Helpers ----------------------------------------------------
        def y_relation(pip_id: int, tip_id: int):
            """Return a condition (pip_id, op, tip_id, 'y', side) based on current pose."""
            pip_y = lms[pip_id]["y"]
            tip_y = lms[tip_id]["y"]
            # In MediaPipe, smaller y = higher on screen
            op = ">" if pip_y > tip_y else "<"
            return (pip_id, op, tip_id, "y", side)

        def x_relation(tip_id: int, neighbor_pip_id: int):
            """Return a condition (tip_id, op, neighbor_pip_id, 'x', side) based on current pose."""
            tip_x = lms[tip_id]["x"]
            neighbor_x = lms[neighbor_pip_id]["x"]
            op = ">" if tip_x > neighbor_x else "<"
            return (tip_id, op, neighbor_pip_id, "x", side)

        # --- Y-axis logic: finger up / down -----------------------------
        # Index finger: landmarks 6 (PIP), 8 (TIP)
        index_pip_id, index_tip_id = 6, 8
        index_finger = lms[index_pip_id:index_tip_id + 1]
        index_is_up = index_finger[0]["y"] > index_finger[2]["y"]  # pip.y > tip.y → tip higher
        index_y_cond = y_relation(index_pip_id, index_tip_id)

        # Middle finger: 10 (PIP), 12 (TIP)
        mid_pip_id, mid_tip_id = 10, 12
        mid_finger = lms[mid_pip_id:mid_tip_id + 1]
        mid_is_up = mid_finger[0]["y"] > mid_finger[2]["y"]
        mid_y_cond = y_relation(mid_pip_id, mid_tip_id)

        # Ring finger: 14 (PIP), 16 (TIP)
        ring_pip_id, ring_tip_id = 14, 16
        ring_finger = lms[ring_pip_id:ring_tip_id + 1]
        ring_is_up = ring_finger[0]["y"] > ring_finger[2]["y"]
        ring_y_cond = y_relation(ring_pip_id, ring_tip_id)

        # Pinky finger: 18 (PIP), 20 (TIP)
        pink_pip_id, pink_tip_id = 18, 20
        pink_finger = lms[pink_pip_id:pink_tip_id + 1]
        pink_is_up = pink_finger[0]["y"] > pink_finger[2]["y"]
        pink_y_cond = y_relation(pink_pip_id, pink_tip_id)

        # Thumb: 2 (MCP/PIP-ish), 4 (TIP)  → open/closed mostly on X axis
        thumb_base_id, thumb_tip_id = 2, 4
        thumb_base = lms[thumb_base_id]
        thumb_tip = lms[thumb_tip_id]
        thumb_is_open = thumb_tip["x"] > thumb_base["x"] if side == "right" else thumb_tip["x"] < thumb_base["x"]
        thumb_x_cond = x_relation(thumb_base_id, thumb_tip_id)  # base vs tip on X

        # --- X-axis logic: separation between fingers ------------------
        x_conds = []

        # For the "V" / spread logic, we compare tips vs neighbor PIPs
        # Example (right hand):
        #   - index tip (8) between thumb (2) and middle PIP (10)
        #   - middle tip (12) between index PIP (6) and ring PIP (14)
        #   - ring tip (16) between middle PIP (10) and pinky PIP (18)
        #   - pinky tip (20) compared with ring PIP (14)

        # Index tip vs thumb base and middle PIP
        x_conds.append(x_relation(index_tip_id, thumb_base_id))  # (8, op, 2, "x", side)
        x_conds.append(x_relation(index_tip_id, mid_pip_id))     # (8, op, 10, "x", side)

        # Middle tip vs index PIP and ring PIP
        x_conds.append(x_relation(mid_tip_id, index_pip_id))     # (12, op, 6, "x", side)
        x_conds.append(x_relation(mid_tip_id, ring_pip_id))      # (12, op, 14, "x", side)

        # Ring tip vs middle PIP and pinky PIP
        x_conds.append(x_relation(ring_tip_id, mid_pip_id))      # (16, op, 10, "x", side)
        x_conds.append(x_relation(ring_tip_id, pink_pip_id))     # (16, op, 18, "x", side)

        # Pinky tip vs ring PIP
        x_conds.append(x_relation(pink_tip_id, ring_pip_id))     # (20, op, 14, "x", side)

        # --- Aggregate logical representation --------------------------
        logical_pose = {
            "hand_side": side,
            "index_finger": "up" if index_is_up else "down",
            "middle_finger": "up" if mid_is_up else "down",
            "ring_finger": "up" if ring_is_up else "down",
            "pink_finger": "up" if pink_is_up else "down",
            "thumb": "open" if thumb_is_open else "closed",
            "conds": [
                index_y_cond,
                mid_y_cond,
                ring_y_cond,
                pink_y_cond,
                thumb_x_cond,
                *x_conds,  # all X-axis relations between neighbors
            ],
        }

        name = self.prompt_gesture_name()
        if name:
            logical_pose["name"] = name
        return logical_pose

    def prompt_gesture_name(self,
        prompt_message: str = "Enter name:",
        window_size=(400, 180),
        font_size=32
    ) -> str | None:
        """
        Opens a small Pygame input window and returns the user-typed string.

        Args:
            prompt_message (str): Text shown above the input box.
            window_size (tuple): Size of the input window (width, height).
            font_size (int): Font size for the input text.

        Returns:
            str | None: 
                The text the user typed (stripped), or None if cancelled.
        """
        if not pygame.get_init():
            pygame.init()

        # Create the text window
        screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("JustCompose - Input")
        font = pygame.font.Font(None, font_size)
        clock = pygame.time.Clock()
        input_box = pygame.Rect(20, 80, 200, 40)
        color_inactive = pygame.Color("lightskyblue3")
        color = color_inactive
        text = ""

        running = True
        while running:
            for event in pygame.event.get():
                # X
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    return None

                if event.type == pygame.KEYDOWN:
                    # ESC
                    if event.key == pygame.K_ESCAPE:
                        pygame.display.quit()
                        return None
                    if event.key == pygame.K_RETURN:
                        typed = text.strip()
                        pygame.display.quit()
                        return typed if typed else None
                    if event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        if event.unicode.isprintable():
                            text += event.unicode

            screen.fill((30, 30, 30)) # Drawing
            prompt_surf = font.render(prompt_message, True, (255, 255, 255)) # Render prompt
            screen.blit(prompt_surf, (20, 30))
            txt_surface = font.render(text, True, color)
            input_box.w = max(200, txt_surface.get_width() + 10) # Resize the box if text gets too long
            screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5)) # Draw text and input box
            pygame.draw.rect(screen, color, input_box, 2)
            pygame.display.flip()
            clock.tick(30)

        return None

    
if __name__ == "__main__":
    recorder = Recorder(name="Just Compose Beta", device=0, capture_mode="landmarks_coords")
    recorder.capture()