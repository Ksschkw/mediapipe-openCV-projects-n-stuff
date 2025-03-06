import cv2 as cv
import mediapipe as mp

vid = cv.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(color=(255,255,0),thickness=1, circle_radius=1)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
               min_tracking_confidence=0.5, max_num_faces=10) as facemesh:

    while True:
        kosi, image = vid.read()
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.flip(image, 1)
        image.flags.writeable= False    
        results = facemesh.process(image)
        # print(results)
        image.flags.writeable= True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list= face_landmarks,
                    connections= mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        cv.imshow('Webcam', image)

        if cv.waitKey(20) & 0xFF==ord('z'):
            break

    vid.release()
    cv.destroyAllWindows()