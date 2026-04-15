import cv2
import time
import numpy as np
from emotion_engine import EmotionEngine

def main():
    # Initialize Emotion Engine (now uses robust Landmarker for detection)
    print("Loading Emotion Engine... (Using AI Landmarker for robust tracking)")
    try:
        engine = EmotionEngine()
    except Exception as e:
        print(f"Error loading engine: {e}")
        return
    
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("System started with Robust Tracking. Press 'q' to quit.")
    
    # Text appearance settings
    font = cv2.FONT_HERSHEY_DUPLEX
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        # Use the upgraded engine to detect both face and reaction
        # This is much more robust to head slants and rotations than Haar Cascades
        result = engine.detect_face_and_reaction(frame)
        
        if result:
            x, y, bw, bh = result["box"]
            reaction = result["reaction"]
            
            # Draw Rectangle
            color = (0, 255, 0) # Green for Active/Happy
            if reaction in ["Sad", "Cry"]: color = (255, 0, 0) # Blue
            elif reaction == "Angry": color = (0, 0, 255) # Red
            elif reaction == "Sleepy": color = (0, 255, 255) # Yellow
            elif reaction in ["Surprise", "Fear"]: color = (255, 255, 0) # Cyan
            
            # Ensure coordinates are within image boundaries for drawing
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            
            # Only draw if the rectangle is visible
            if x2 > x1 and y2 > y1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw Label Background
                label_y = max(40, y1)
                cv2.rectangle(frame, (x1, label_y - 40), (x2, label_y), color, -1)
                cv2.putText(frame, f"{reaction}", (x1 + 5, label_y - 10), font, 0.8, (255, 255, 255), 1)

        # Show Output
        cv2.imshow('Face Reaction System - AI Robust Mode', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
