import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf

# 0. GPU가 잘 잡히는지 확인
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"🔥🔥 GPU 가동 확인! (RTX 3060이 일을 시작합니다): {gpus}")
else:
    print("⚠️ GPU를 못 찾았습니다. CPU로 학습합니다.")

# 1. 데이터 수집 (삼성전자 10년치)
print("데이터 다운로드 중...")
df = yf.Ticker("005930.KS").history(period="10y")
data = df['Close'].values.reshape(-1, 1) # 종가만 가져옴

# 2. 데이터 정규화 (0~1 사이 숫자로 변환)
# AI는 큰 숫자(70,000원)보다 작은 숫자(0.7)를 더 잘 계산합니다.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 3. 학습용 데이터셋 만들기 (50일치 보고 다음날 맞추기)
window_size = 50  # 과거 50일을 보고
X = []
y = []

for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i-window_size:i, 0]) # 0~49일 데이터 (문제)
    y.append(scaled_data[i, 0])               # 50번째 날 데이터 (정답)

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # LSTM이 좋아하는 형태로 변형

# 데이터를 훈련용(80%)과 테스트용(20%)으로 나누기
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. LSTM 모델 설계 (AI 뇌 구조 만들기)
print("AI 모델 빌드 중...")
model = Sequential()
# 50개의 기억 뉴런을 가진 LSTM 층
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1)) # 결과는 '내일 주가' 숫자 딱 하나

model.compile(optimizer='adam', loss='mean_squared_error')

# 5. 학습 시작! (여기가 하이라이트)
print("학습 시작! (터미널의 게이지를 보세요)")
model.fit(X_train, y_train, epochs=20, batch_size=32)

# 6. 예측 및 결과 시각화
print("예측 진행 중...")
predictions = model.predict(X_test)
# 0~1로 압축했던 숫자를 다시 원래 가격(원화)으로 되돌리기
predictions = scaler.inverse_transform(predictions)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# 그래프 그리기
plt.figure(figsize=(12, 6))
plt.plot(y_test_real, color='blue', label='Actual Price (Samsung)')
plt.plot(predictions, color='red', label='AI Predicted Price')
plt.title('Samsung Electronics Price Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.savefig('lstm_result.png')
print("✅ 완료! 'lstm_result.png' 파일을 확인해보세요.")

# ==========================================
# 7. 드디어 "내일 주가" 예측하기
# ==========================================
print("\n🔮 AI가 내일 주가를 분석 중입니다...")

# 최근 50일치 데이터를 가져옵니다
last_50_days = data[-window_size:] 

# AI가 읽을 수 있게 0~1 사이로 변환 (정규화)
last_50_days_scaled = scaler.transform(last_50_days)

# 모양을 맞춰줍니다 (1개 데이터, 50일치, 1개 특성)
X_tomorrow = last_50_days_scaled.reshape(1, window_size, 1)

# 예측하기!
predicted_price_scaled = model.predict(X_tomorrow)

# 0~1로 나온 결과를 다시 '원화(KRW)'로 변환
predicted_price = scaler.inverse_transform(predicted_price_scaled)

# 삼성전자 객체 다시 소환
samsung = yf.Ticker("005930.KS")

# 최신 현재가(장중이면 실시간, 장 마감이면 종가) 가져오기
current_price = samsung.fast_info['last_price']

print("=" * 30)
print(f"📉 오늘 삼성전자 종가 : {current_price:,.0f}원")
print(f"🔮 AI 예측 내일 주가  : {predicted_price[0][0]:,.0f}원")
print("=" * 30)