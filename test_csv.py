import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp


def calculate_angles(hand_landmarks):
    joint = np.zeros((21, 3))
    for j, lm in enumerate(hand_landmarks.landmark):
        joint[j] = [lm.x, lm.y, lm.z]

    # Compute angles between joints
    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
    v = v2 - v1  # [20,3]
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

    angle = np.degrees(angle)  # Convert radian to degree
    return angle

# 손 동작 분류를 위한 모델 불러오기
model = load_model('gesture_detection_model.h5')

# 미디어파이프 손 감지 모듈 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 웹캠 초기화
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 영상 반전
    frame = cv2.flip(frame, 1)
    
    # 영상 처리를 위해 BGR을 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 손 감지 수행
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 손 감지 결과를 표시하기 위해 랜드마크를 그림
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 손 감지 결과를 모델에 입력할 형태로 가공
            angle = calculate_angles(hand_landmarks)
            input_data = np.array(angle).reshape(1, -1)
            
            # 모델로 손 동작 분류
            prediction = model.predict(input_data)
            gesture_label = int(np.round(prediction)[0])
            
            # 분류 결과에 따라 텍스트 표시
            if gesture_label == 0:
                cv2.putText(frame, 'Right', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Left', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 영상 출력
    cv2.imshow('Gesture Detection', frame)
    
    # 종료 조건 설정
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 해제
cap.release()
cv2.destroyAllWindows()