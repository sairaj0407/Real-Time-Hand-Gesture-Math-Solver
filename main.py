import cv2
import mediapipe as mp
import time
from collections import deque

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

expression = ""
result = ""
last_detect_time = 0
cooldown = 1.5  # seconds

# Gesture buffer for stability
gesture_buffer = deque(maxlen=7)

# One-hand gesture mapping
gesture_map = {
    1: "1",
    2: "2",
    3: "3",
    4: "+",
    5: "-"
}

# Finger counting
def count_fingers(hand_landmarks, hand_label):
    finger_tips = [4, 8, 12, 16, 20]
    finger_pips = [3, 6, 10, 14, 18]
    count = 0

    # Thumb
    thumb_tip = hand_landmarks.landmark[4]
    thumb_ip = hand_landmarks.landmark[3]

    if hand_label == "Right":
        if thumb_tip.x < thumb_ip.x - 0.02:
            count += 1
    else:
        if thumb_tip.x > thumb_ip.x + 0.02:
            count += 1

    # Other fingers
    for tip, pip in zip(finger_tips[1:], finger_pips[1:]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1

    return count

# Gesture interpretation
def interpret_gesture(counts, total_hands):
    global expression, result, last_detect_time

    current_time = time.time()
    if current_time - last_detect_time < cooldown:
        return

    # -------- TWO HAND GESTURES --------
    if total_hands == 2:
        left, right = counts

        # Clear
        if left == 5 and right == 5:
            expression = ""
            result = ""

        # Multiply
        elif left == 2 and right == 2:
            if expression and expression[-1] not in "+-*/":
                expression += "*"

        # Divide
        elif left == 3 and right == 3:
            if expression and expression[-1] not in "+-*/":
                expression += "/"

        last_detect_time = current_time
        gesture_buffer.clear()
        return

    # -------- ONE HAND GESTURES --------
    val = counts[0]
    gesture_buffer.append(val)

    if gesture_buffer.count(val) < 6:
        return

    # Evaluate
    if val == 0:
        try:
            result = str(eval(expression))
            expression = result
        except:
            result = "Error"
            expression = ""

    elif val in gesture_map:
        symbol = gesture_map[val]

        if symbol in "+-" and expression.endswith("+-*/"):
            return

        expression += symbol

    last_detect_time = current_time
    gesture_buffer.clear()

# Start webcam
cap = cv2.VideoCapture(0)
time.sleep(1.5)  # Camera warm-up

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    finger_counts = []

    if results.multi_hand_landmarks:
        for idx, hand_landmark in enumerate(results.multi_hand_landmarks):
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS)

            hand_label = results.multi_handedness[idx].classification[0].label
            fingers = count_fingers(hand_landmark, hand_label)
            finger_counts.append(fingers)

            cv2.putText(
                img, f"Fingers: {fingers}",
                (10, 120 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 0), 2
            )

        interpret_gesture(finger_counts, len(finger_counts))
    else:
        gesture_buffer.clear()

    # UI
    cv2.rectangle(img, (0, 0), (700, 90), (0, 0, 0), -1)
    cv2.putText(img, f"Expression: {expression}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Result: {result}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(
        img,
        "1/2/3 Digits | 4:+ | 5:- | 0:= | (2+2)* | (3+3)/ | (5+5) Clear | ESC Exit",
        (5, 460),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        (100, 255, 255), 1
    )

    cv2.imshow("🖐️ Real-Time Hand Gesture Math Solver", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
