import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Finger tip landmarks.
TIP_IDS = [4, 8, 12, 16, 20]

def fingers_up(hand_landmarks):
    """
    Determines which fingers are up.
    Returns a list of 0/1 indicating if each finger is up.
    Thumb: comparing tip.x and ip.x since thumb points sideways.
    Other fingers: tip.y < pip.y means finger is up.
    """
    fingers = []

    # Thumb
    if hand_landmarks.landmark[TIP_IDS[0]].x < hand_landmarks.landmark[TIP_IDS[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for tip_id in TIP_IDS[1:]:
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[tip_id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

def recognize_gesture(fingers):
    """
    Recognizes gesture based on fingers up.
    """
    total_fingers = sum(fingers)
    if total_fingers == 0:
        return 'FIST'
    elif total_fingers == 5:
        return 'PALM'
    else:
        return 'UNKNOWN'

def main():
    cap = cv2.VideoCapture(0)
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Game variables
    box_size = 50
    box_x = width // 2 - box_size // 2
    box_y = height // 2 - box_size // 2
    box_color = (0, 255, 0)
    move_speed = 15

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            frame = cv2.flip(frame, 1)  # Mirror image for natural interaction
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            gesture = "None"
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                fingers = fingers_up(hand_landmarks)
                gesture = recognize_gesture(fingers)

                # Move the box based on gesture
                if gesture == 'PALM':
                    box_y = max(0, box_y - move_speed)
                elif gesture == 'FIST':
                    box_y = min(height - box_size, box_y + move_speed)

            # Draw the game box
            cv2.rectangle(frame, (box_x, box_y), (box_x+box_size, box_y+box_size), box_color, -1)
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.putText(frame, 'Show PALM to move UP, FIST to move DOWN', (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

            cv2.imshow('Hand Gesture Game', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
