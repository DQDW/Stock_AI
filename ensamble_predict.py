import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import os
import time  # <--- 시간을 재기 위한 시계 모듈 추가

# 불필요한 로그 제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. 설정
# ==========================================
targets = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "현대차": "005380.KS",
    "NAVER": "035420.KS"
}

ENSEMBLE_COUNT = 5     # 5명의 AI 위원회
EPOCHS = 100           # 100번 반복 학습
BATCH_SIZE = 64        # 3060 가속
DATA_PERIOD = "20y"    # 20년치 데이터

results = []

def predict_stock_ensemble(name, ticker):
    print(f"\n🔄 [{name}] 20년치 데이터 로딩 중... AI 학습 시작!")
    
    try:
        df = yf.Ticker(ticker).history(period=DATA_PERIOD)
        if len(df) < 100: 
            print(f"⚠️ {name}: 데이터 부족.")
            return None
            
        current_price = df['Close'].iloc[-1]
        data = df['Close'].values.reshape(-1, 1)
        print(f"   📊 학습 데이터: 총 {len(df)}일치 확보")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        window_size = 50
        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        predictions = []
        
        for i in range(ENSEMBLE_COUNT):
            model = Sequential()
            model.add(LSTM(50, return_sequences=False, input_shape=(window_size, 1)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
            
            last_50 = scaled_data[-window_size:].reshape(1, window_size, 1)
            pred_scaled = model.predict(last_50, verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            
            predictions.append(pred_price)
            print(f"   👉 [{i+1}/{ENSEMBLE_COUNT}] 예측: {pred_price:,.0f}원", end="\r")

        print(f"\n   ✅ 평균 예측 완료!")

        avg_price = np.mean(predictions)
        gap = avg_price - current_price
        rate = (gap / current_price) * 100
        direction = "🔺상승" if gap > 0 else "🔽하락"

        return {
            "종목명": name,
            "현재가": f"{current_price:,.0f}원",
            "내일예측(평균)": f"{avg_price:,.0f}원",
            "예상등락": f"{gap:+,.0f}원 ({rate:+.2f}%)",
            "방향": direction
        }

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return None

# ==========================================
# 실행부 (시간 측정 시작)
# ==========================================
start_time = time.time()  # <--- 스톱워치 시작!

print("=" * 60)
print(f"🚀 [ULTIMATE 모드] 20년 데이터 x 100회 학습 (RTX 3060)")
print("=" * 60)

for name, ticker in targets.items():
    result = predict_stock_ensemble(name, ticker)
    if result:
        results.append(result)

# ==========================================
# 결과 출력 및 시간 계산
# ==========================================
end_time = time.time()       # <--- 스톱워치 멈춤!
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("\n" + "=" * 65)
print("📊 [AI 주가 예측 최종 리포트 (20년 데이터 기반)]")
print("=" * 65)
df_result = pd.DataFrame(results)
df_result = df_result[['종목명', '현재가', '내일예측(평균)', '예상등락', '방향']]
print(df_result.to_string(index=False))
print("=" * 65)
print(f"⏱️ 총 소요 시간: {minutes}분 {seconds}초") # <--- 여기 출력됨
print("=" * 65)