# Responsible Camera for Motion Capture
import cv2 as cv
import mediapipe as mp
import pygame
import time


class Camera:
    # Handling with the capture from the source
    # to then send the landmark to an analyzer class
    
    def __init__(self, name="Just Compose Beta", device=0):
        # Pygame structure
        self.mixer = pygame.mixer
        self.mixer.init()
        self.mixer.music.load("C:/Users/lusca/Universidade/CV/TPs/TPFinal/JustCompose/assets/boing.mp3")

        # MediaPipe structure
        self.mp_hands = mp.solutions.hands 
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Class attributes
        self.name = name
        self.device = device
        self.compatible_file_types = ('.jpg', '.jpeg', '.png')
        
        # Start the capture (the main function of the class)
        self.capture()
    
    def capture(self):
        if self.device == 0:        
            cap = cv.VideoCapture(self.device)

            with self.mp_hands.Hands() as hand_detector:
                CAPTURE = cv.VideoCapture(self.device)
                if not CAPTURE.isOpened(): 
                    print("No video source :(")
                    exit(1)
                    
                while CAPTURE.isOpened():
                    ret, frame = CAPTURE.read()
                    if not ret: # frame not captured
                        print("The video has no frames")
                        
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
            self.mp_drawing.draw_landmarks(
                image, 
                hand_landmarks, 
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(235, 137, 52), thickness=1, circle_radius=3),
                self.mp_drawing.DrawingSpec(color=(235, 52, 113), thickness=1, circle_radius=1)
            )
            # 1º arg: image to draw on
            # 2º arg: landmarks to draw
            # 3º arg: the connections between the landmarks
            # 4º arg: style for the circles (landmarks)
            # ==========================================================
            
            coord = tuple(
                (int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x * image.shape[1]) - 30,
                 int(hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].y * image.shape[0]) + 30)
            )
            label = handedness.classification[0].label
            if label == "Left":
                if self.mixer.music.get_busy():
                    continue # already playing
                self.mixer.music.play()
                time.sleep(1)
                self.mixer.music.stop()
