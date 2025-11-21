# Responsible Camera for Motion Capture
import cv2 as cv
import mediapipe as mp


class Camera:
    # Handling with the capture from the source
    # to then send the landmark to an analyzer class
    
    def __init__(self, name="Just Compose Beta"):
        self.mp_hands = mp.solutions.hands 
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera source TODO: change to a variable input
        cap = cv.VideoCapture(0)
        
        with self.mp_hands.Hands() as hand_detector:
            while cv.pollKey() == -1:
                success, frame = cap.read()
                if not success:
                    print("Nothing to see here!!")
                    continue
                
                # Make detections
                results = hand_detector.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    self.draw_landmarks(frame, results)
                
                cv.imshow(name, frame)
            
            cap.release()
            cv.destroyAllWindows()
            
    def draw_landmarks(self, image, results):
        for hand_landmarks in results.multi_hand_landmarks:
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