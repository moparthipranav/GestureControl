import cv2
import mediapipe as mp
import pyautogui
import pygetwindow as gw
import time

# -----------------------------
# Function to focus Chrome window
# -----------------------------
def focus_browser():
    windows = gw.getWindowsWithTitle("Subway Surfers")  # Use game tab title if possible
    if windows:
        win = windows[0]
        win.activate()
        time.sleep(0.2)  # small delay to ensure focus

# -----------------------------
# Initialize MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Open webcam with lower resolution
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# -----------------------------
# Variables
# -----------------------------
last_gesture = None

# -----------------------------
# Focus browser once
# -----------------------------
focus_browser()

# -----------------------------
# Main loop
# -----------------------------
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    gesture = "None"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # -----------------------------
            # Index finger tip position
            # -----------------------------
            index_tip = hand_landmarks.landmark[8]  # 8 = index finger tip
            finger_x, finger_y = int(index_tip.x * w), int(index_tip.y * h)

            # Frame center
            center_x, center_y = w // 2, h // 2
            offset_x, offset_y = 40, 30  # smaller tolerance for faster response

            # Determine gesture based on index finger tip
            if finger_x < center_x - offset_x:
                gesture = "LEFT"
            elif finger_x > center_x + offset_x:
                gesture = "RIGHT"
            elif finger_y < center_y - offset_y:
                gesture = "UP"
            elif finger_y > center_y + offset_y:
                gesture = "DOWN"
            else:
                gesture = "CENTER"

            # Draw markers
            cv2.circle(frame, (finger_x, finger_y), 5, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), cv2.FILLED)

            # Show gesture text
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Map gestures to keyboard keys (avoid repeats)
            if gesture != last_gesture:
                if gesture == "LEFT":
                    pyautogui.press("left")
                elif gesture == "RIGHT":
                    pyautogui.press("right")
                elif gesture == "UP":
                    pyautogui.press("up")
                elif gesture == "DOWN":
                    pyautogui.press("down")
                last_gesture = gesture

    # Show webcam frame
    cv2.imshow("Hand Gesture Control", frame)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Release resources
# -----------------------------
cap.release()
cv2.destroyAllWindows()
