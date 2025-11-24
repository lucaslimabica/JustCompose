# Responsible Camera for Motion Capture
import cv2 as cv
import mediapipe as mp
import pygame


class Camera:
    # Handling with the capture from the source
    # to then send the landmark to an analyzer class
    
    def __init__(self, name="Just Compose Beta", device=0, capture_mode=None):
        """TODO _summary_

        Args:
            name (str, optional): _description_. Defaults to "Just Compose Beta".
            device (int, optional): _description_. Defaults to 0.
            capture_mode (str, optional): Determines wich text will be displayed on the screen.
            Options -> landmarks, landmarks_names, landmarks_coords, None. Defaults to None.
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
        if self.device == 0:        
            with self.mp_hands.Hands() as hand_detector:
                self.cap = cv.VideoCapture(self.device)
                if not self.cap.isOpened(): 
                    print("No video source :(")
                    exit(1)
                    
                while self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if not ret: # frame not captured
                        print("The video has no frames")
                    
                    # Mirror the frame
                    frame = cv.flip(frame, 1)
                    # Make detections
                    results = hand_detector.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                    if results.multi_hand_landmarks: # Avoid None for the drawing func
                        # Call the drawing func
                        self.draw_landmarks(frame, results)
                    cv.imshow(self.name, frame)
                    
                    key = cv.waitKey(1)
                    if key in [27, ord("q"), ord("l")]:
                        break
                
        elif isinstance(self.device, str) and self.device.endswith(self.compatible_file_types):
            image = cv.imread(self.device)

            with self.mp_hands.Hands(static_image_mode=True) as hand_detector:
                # Make detections
                results = hand_detector.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
                if results.multi_hand_landmarks: # Avoid None for the drawing func
                    # Call the drawing func
                    self.draw_landmarks(image, results)

                cv.imshow(self.name, image)
                cv.waitKey(0)
                cv.destroyAllWindows()
                
    def draw_landmarks(self, image, results):
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
            self.draw_landmark_names(image, hand_landmarks)
            self.draw_bounding_box(image, hand_landmarks.landmark)
            
    def draw_landmark_names(self, image, hand_landmarks):
        for i, landmark in enumerate(hand_landmarks.landmark):
                # Depending on the capture mode, display differents texts,
                # perfect for debugging and development of gesture recognition
                coords = ""
                if self.capture_mode == "landmarks_coords":
                    coords = f"{i}: ({landmark.x:.2f}, {landmark.y:.2f})"
                elif self.capture_mode == "landmarks":
                    coords = f"{i}"
                    
                width  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                px = int(landmark.x * width)
                py = int(landmark.y * height)
                cv.putText(img=image, text=coords, org=(px + 30, py), fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=(0, 0, 0), lineType = cv.LINE_AA)
            
    def draw_bounding_box(self, image, hand_landmarks):
        # Calculate bounding box
        # the most left, right, top and bottom points are based on the landmark
        # 0 is the hand base, 4 is the thumb tip, 8 is the index finger tip, 12 is the middle finger tip, 16 is the ring finger tip, 20 is the pinky tip
        width  = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        x1 = int(hand_landmarks[4].x * width) - 20
        y1 = int(hand_landmarks[12].y * height) - 20
        x2 = int(hand_landmarks[20].x * width) + 20
        y2 = int(hand_landmarks[0].y * height) + 20
        cv.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=2)