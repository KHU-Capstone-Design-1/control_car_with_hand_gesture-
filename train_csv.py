import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 데이터셋 불러오기
dataset = pd.read_csv('data\gesture_dataset.csv', header=None)

# 입력 데이터와 타깃 데이터로 분리
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values



# # 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 훈련 세트와 테스트 세트로 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 구축
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(X_train, y_train, epochs=2500, batch_size=32, validation_data=(X_test, y_test))

# 모델 저장
model.save('gesture_detection_model.h5')

