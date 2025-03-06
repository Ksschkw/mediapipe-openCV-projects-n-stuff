import cv2 as cv
import mediapipe as mp

img=cv.VideoCapture(0)
mp_draw=mp.solutions.drawing_utils
mp_hands=mp.solutions.hands

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2) as hands:
    while True:
        ret, frame = img.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.flip(frame, 1)
        frame.flags.writeable = False
        results=hands.process(frame)    
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,0,255),thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(255,0,0),thickness=2, circle_radius=2))
        cv.imshow('Webcam', frame)
        if cv.waitKey(20) & 0xFF==ord('z'):
            break   
img.release()
cv.destroyAllWindows()