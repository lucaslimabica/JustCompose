# Responsible Camera for Motion Capture
import cv2 as cv
import mediapipe as mp


class Camera:
    # Handling with the capture from the source
    # to then send the landmark to an analyzer class
    
    def __init__(self, name="MediaPipe Result"):
        self.mp_holistic = mp.solutions.holistic 
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera source TODO: change to a variable input
        cap = cv.VideoCapture(0)
        
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            mediapipe_detection = lambda image: holistic.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
            
            while cv.pollKey() == -1:
                success, frame = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue
                
                # Make detections
                results = mediapipe_detection(frame)
                
                self.draw_landmarks(frame, results)
                
                cv.imshow(name, frame)
            
            cap.release()
            cv.destroyAllWindows()
            
    def draw_landmarks(self, image, results):
        # Draw face connections
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
        self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        
        # Draw pose connections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
        self.mp_drawing.DrawingSpec(color=(80, 22, 255), thickness=1, circle_radius=3),
        self.mp_drawing.DrawingSpec(color=(80, 44, 255), thickness=1, circle_radius=1))
        
        # Draw left hand connections
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
        self.mp_holistic.HAND_CONNECTIONS, self.mp_drawing.DrawingSpec(color=(255, 22, 76),
        thickness=1, circle_radius=3), self.mp_drawing.DrawingSpec(color=(255, 44, 250),
        thickness=1, circle_radius=1))
        
        # Draw right hand connections
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
        self.mp_holistic.HAND_CONNECTIONS, self.mp_drawing.DrawingSpec(color=(245, 255, 66),
        thickness=1, circle_radius=3), self.mp_drawing.DrawingSpec(color=(245, 255, 230),
        thickness=1, circle_radius=1))