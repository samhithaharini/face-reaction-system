import cv2
import numpy as np
from fer.fer import FER
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

class EmotionEngine:
    def __init__(self):
        # Initialize FER detector
        self.detector = FER(mtcnn=False) 
        
        # Initialize MediaPipe Tasks Face Landmarker
        self.use_mediapipe = False
        model_path = 'face_landmarker.task'
        
        if os.path.exists(model_path):
            try:
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=True,
                    num_faces=1
                )
                self.landmarker = vision.FaceLandmarker.create_from_options(options)
                self.use_mediapipe = True
                print("MediaPipe Landmarker initialized.")
            except Exception as e:
                print(f"Warning: Could not initialize MediaPipe Landmarker: {e}")
        
        # EAR threshold for sleepy detection
        self.EAR_THRESHOLD = 0.22
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def calculate_ear(self, landmarks, eye_indices):
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
        horiz = np.linalg.norm(p1 - p4)
        vert1 = np.linalg.norm(p2 - p6)
        vert2 = np.linalg.norm(p3 - p5)
        return (vert1 + vert2) / (2.0 * horiz)

    def detect_face_and_reaction(self, frame):
        """Detects face and reaction in the whole frame, returning bbox and reaction."""
        if not self.use_mediapipe:
            return None
            
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        detection_result = self.landmarker.detect(mp_image)
        
        if not detection_result.face_landmarks:
            return None

        # 1. Get Bounding Box from Landmarks (robust to rotation/slant)
        landmarks = detection_result.face_landmarks[0]
        coords = np.array([(lm.x * w, lm.y * h) for lm in landmarks])
        
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        
        # Add some padding
        pad = 0.2
        bw, bh = x_max - x_min, y_max - y_min
        x, y = int(x_min - pad*bw), int(y_min - pad*bh)
        bw, bh = int(bw * (1 + 2*pad)), int(bh * (1 + 2*pad))
        
        # 2. Get standard emotions for the face ROI
        roi_x, roi_y = max(0, x), max(0, y)
        roi_w, roi_h = min(w - roi_x, bw), min(h - roi_y, bh)
        face_roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        primary_emotion = "neutral"
        score = 0
        if face_roi.size > 0:
            res = self.detector.detect_emotions(face_roi)
            if res:
                emotions = res[0]["emotions"]
                primary_emotion = max(emotions, key=emotions.get)
                score = emotions[primary_emotion]

        # 3. Refine with Blendshapes and Landmarks
        reaction = primary_emotion.capitalize()
        
        # Get EAR for Sleepy
        left_ear = self.calculate_ear(coords, self.LEFT_EYE)
        right_ear = self.calculate_ear(coords, self.RIGHT_EYE)
        ear_val = (left_ear + right_ear) / 2.0
        
        # Custom logic for Sleepy
        if ear_val < self.EAR_THRESHOLD:
            reaction = "Sleepy"
            
        # Custom logic for "Active"
        elif reaction == "Neutral" and ear_val > 0.24:
            reaction = "Active"

        # Refine Anger using Blendshapes (more sensitive to brow movement)
        if detection_result.face_blendshapes:
            blendshapes = {b.category_name: b.score for b in detection_result.face_blendshapes[0]}
            
            # browDown represents anger/concentration
            # browInnerUp + eyeWideOpen represents surprise
            
            brow_down = (blendshapes.get('browDownLeft', 0) + blendshapes.get('browDownRight', 0)) / 2.0
            brow_up = blendshapes.get('browInnerUp', 0)
            
            # If eyebrows are strongly down, it's very likely Anger/Angry
            if brow_down > 0.4:
                reaction = "Angry"
            # If user thinks "raised brows" is anger, let's check high browUp + maybe mouth tension?
            # But usually raised brows is Surprise.
            elif brow_up > 0.5 and reaction != "Happy":
                 # If only brow is up, might be user's version of intense focus/anger
                 # We'll stick to Angry if browDown is high, but Surprise if browUp is high.
                 if reaction != "Angry": reaction = "Surprise"

        # Map Sad to Cry
        if reaction == "Sad" and score > 0.55:
            reaction = "Cry"

        return {
            "box": (x, y, bw, bh),
            "reaction": reaction,
            "score": score
        }
