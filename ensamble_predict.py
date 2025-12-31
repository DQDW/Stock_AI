import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import os
import time
import datetime

# 불필요한 로그 제거
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. 설정 (종목 추가 완료!)
# ==========================================
targets = {
    # 🇰🇷 한국 주식
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "방림": "003610.KS",          # (추가됨)
    "강원에너지": "114190.KQ",      # (추가됨 - 코스닥)
    
    # 🇺🇸 미국 주식
    "록히드마틴": "LMT",           # (추가됨)
    "보잉": "BA"                  # (추가됨)
}

ENSEMBLE_COUNT = 5     # 5번 반복 학습
EPOCHS = 100           # 100번 학습
BATCH_SIZE = 64        # 3060 성능 활용
DATA_PERIOD = "20y"    # 20년치 데이터

results = []

def predict_stock_ensemble(name, ticker):
    print(f"\n🔄 [{name}] 20년치 데이터 로딩 중... AI 학습 시작!")
    
    # 화폐 단위 결정 (티커에 .KS나 .KQ가 없으면 미국 주식으로 간주)
    if ".KS" in ticker or ".KQ" in ticker:
        currency = "원"
        is_korea = True
    else:
        currency = "달러"
        is_korea = False

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
            
            # 출력할 때 한국은 소수점 없이, 미국은 소수점 2자리까지
            if is_korea:
                print(f"   👉 [{i+1}/{ENSEMBLE_COUNT}] 예측: {pred_price:,.0f}{currency}", end="\r")
            else:
                print(f"   👉 [{i+1}/{ENSEMBLE_COUNT}] 예측: {pred_price:,.2f}{currency}", end="\r")

        print(f"\n   ✅ 평균 예측 완료!")

        avg_price = np.mean(predictions)
        gap = avg_price - current_price
        rate = (gap / current_price) * 100
        direction = "🔺상승" if gap > 0 else "🔽하락"

        # 결과 포맷팅 (한국: 정수 / 미국: 소수점)
        if is_korea:
            price_str = f"{current_price:,.0f}{currency}"
            pred_str = f"{avg_price:,.0f}{currency}"
            gap_str = f"{gap:+,.0f}{currency}"
        else:
            price_str = f"{current_price:,.2f}{currency}"
            pred_str = f"{avg_price:,.2f}{currency}"
            gap_str = f"{gap:+,.2f}{currency}"

        return {
            "종목명": name,
            "현재가": price_str,
            "내일예측(평균)": pred_str,
            "예상등락": f"{gap_str} ({rate:+.2f}%)",
            "방향": direction
        }

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        return None

# ==========================================
# 실행 및 시간 측정
# ==========================================
start_time = time.time()

print("=" * 60)
print(f"🚀 [글로벌 모드] 한국/미국 주식 통합 분석 (RTX 3060)")
print("=" * 60)

for name, ticker in targets.items():
    result = predict_stock_ensemble(name, ticker)
    if result:
        results.append(result)

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)

print("\n" + "=" * 65)
print("📊 [AI 주가 예측 최종 리포트]")
print("=" * 65)

df_result = pd.DataFrame(results)
df_result = df_result[['종목명', '현재가', '내일예측(평균)', '예상등락', '방향']]
print(df_result.to_string(index=False))
print("=" * 65)
print(f"⏱️ 총 소요 시간: {minutes}분 {seconds}초")

# CSV 파일 저장
today_str = datetime.datetime.now().strftime("%Y-%m-%d")
filename = f"stock_prediction_{today_str}.csv"
df_result.to_csv(filename, index=False, encoding='utf-8-sig')

print(f"💾 결과 저장 완료: {filename}")
print("=" * 65)