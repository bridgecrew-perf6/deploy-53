import mediapipe as mp
import cv2
import numpy as np
import uuid
import os


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def output():
    if not os.path.exists('Output_Images'):
        os.makedirs('Output_Images')

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        while cap.isOpened():
            ret, frame = cap.read()

            # BGR 2 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Flip on horizontal
            image = cv2.flip(image, 1)

            # Set flag
            image.flags.writeable = False

            # Detections
            results = hands.process(image)

            # Set flag to true
            image.flags.writeable = True

            # RGB 2 BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Detections
            print(results)

            # Rendering results
            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,
                                                                     circle_radius=2),
                                              )

            # Save our image
            cv2.imwrite(os.path.join('Output_Images', '{}.jpg'.format(uuid.uuid1())), image)
            cv2.imshow('Hand Tracking', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def detect_multihand():
    image = cv2.imread('Output_Images/WIN_20220616_15_02_35_Pro.jpg')

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=2) as hands:
        # BGR 2 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Flip on horizontal
        image = cv2.flip(image, 1)

        # Set flag
        image.flags.writeable = False

        # Detections
        results = hands.process(image)

        # Set flag to true
        image.flags.writeable = True

        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        print(results.multi_hand_landmarks)
        # print(results.multi_hand_world_landmarks)
        print(results.multi_handedness)


if __name__ == '__main__':
    detect_multihand()
