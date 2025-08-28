import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

expression = ""
result = ""
last_detect_time = 0
cooldown = 1.5  # in seconds

# Gesture mapping
gesture_map = {
    0: "=",   # Evaluate
    1: "1",
    2: "2",
    3: "3",
    4: "+",
    5: "-"
}

# Improved finger count function
def count_fingers(hand_landmarks, hand_label):
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [2, 6, 10, 14, 18]
    count = 0

    # Thumb (Left or Right handedness)
    if hand_label == "Right":
        if hand_landmarks.landmark[finger_tips[0]].x < hand_landmarks.landmark[finger_pips[0]].x:
            count += 1
    else:  # Left
        if hand_landmarks.landmark[finger_tips[0]].x > hand_landmarks.landmark[finger_pips[0]].x:
            count += 1

    # Other fingers (y-based)
    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1

    return count

# Logic to interpret gesture
def interpret_gesture(counts, total_hands):
    global expression, result, last_detect_time
    current_time = time.time()

    if current_time - last_detect_time < cooldown:
        return

    if total_hands == 2:
        if counts == [5, 5]:
            expression = ""
            result = ""
        elif counts == [1, 1]:
            print("Exiting...")
            exit(0)
        last_detect_time = current_time

    elif total_hands == 1:
        val = counts[0]
        if val in gesture_map:
            symbol = gesture_map[val]
            if symbol == "=":
                try:
                    result = str(eval(expression))
                    expression = result
                except:
                    result = "Error"
                    expression = ""
            else:
                expression += symbol
            last_detect_time = current_time

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_count = 0
    finger_counts = []

    if results.multi_hand_landmarks:
        for idx, hand_landmark in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

            hand_label = results.multi_handedness[idx].classification[0].label  # "Left" or "Right"
            fingers = count_fingers(hand_landmark, hand_label)

            finger_counts.append(fingers)
            hand_count += 1

        interpret_gesture(finger_counts, hand_count)

    # Draw overlay
    cv2.rectangle(img, (0, 0), (700, 80), (0, 0, 0), -1)
    cv2.putText(img, f"Expression: {expression}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Result: {result}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Show gesture instructions
    cv2.putText(img, "1/2/3 = Digits | 4:+ | 5:- | 0:= | (5+5)=Clear | (1+1)=Exit",
                (5, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 255), 1)

    # Show window
    cv2.imshow("🖐️ Real-Time Hand Gesture Math Solver", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

