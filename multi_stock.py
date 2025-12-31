import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import os

# GPU 경고 메시지 끄기 (깔끔하게 보기 위함)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==========================================
# 1. 분석하고 싶은 종목 리스트 (이름: 코드)
# ==========================================
targets = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "현대차": "005380.KS",
    "NAVER": "035420.KS"
}

# 결과를 담을 보따리
results = []

def predict_stock(name, ticker):
    print(f"\n🔄 [{name}] 데이터 분석 및 AI 학습 시작...")
    
    # 1. 데이터 수집 (
    try:
        df = yf.Ticker(ticker).history(period="10y") #날짜 설정
        if len(df) < 100:
            print(f"⚠️ {name}: 데이터가 너무 적어서 건너뜁니다.")
            return None
            
        current_price = df['Close'].iloc[-1] # 오늘 종가
        
        data = df['Close'].values.reshape(-1, 1)

        # 2. 정규화
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # 3. 데이터셋 만들기
        window_size = 50
        X, y = [], []
        for i in range(window_size, len(scaled_data)):
            X.append(scaled_data[i-window_size:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        # 4. 모델 학습 (verbose=0으로 설정해서 지저분한 로그 끔)
        model = Sequential()
        model.add(LSTM(50, return_sequences=False, input_shape=(window_size, 1)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # 학습 (종목당 10번만 반복 - 속도 조절)
        model.fit(X, y, epochs=20, batch_size=32, verbose=0) 

        # 5. 내일 가격 예측
        last_50_days = scaled_data[-window_size:].reshape(1, window_size, 1)
        predicted_scaled = model.predict(last_50_days, verbose=0)
        predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

        # 6. 결과 정리
        gap = predicted_price - current_price
        rate = (gap / current_price) * 100
        
        direction = "🔺상승" if gap > 0 else "🔽 하락"
        
        print(f"✅ [{name}] 분석 완료! 예측가: {predicted_price:,.0f}원 ({direction})")
        
        return {
            "종목명": name,
            "현재가": f"{current_price:,.0f}원",
            "내일예측": f"{predicted_price:,.0f}원",
            "예상등락": f"{gap:+,.0f}원 ({rate:+.2f}%)",
            "방향": direction
        }

    except Exception as e:
        print(f"❌ {name} 처리 중 에러 발생: {e}")
        return None

# ==========================================
# 2. 반복문으로 전체 실행
# ==========================================
print("=" * 50)
print(f"🚀 총 {len(targets)}개 종목 분석을 시작합니다 (GPU 가속)")
print("=" * 50)

for name, ticker in targets.items():
    result = predict_stock(name, ticker)
    if result:
        results.append(result)

# ==========================================
# 3. 최종 결과표 출력
# ==========================================
print("\n" + "=" * 60)
print("📊 [AI 주가 예측 최종 리포트]")
print("=" * 60)

# 보기 좋게 데이터프레임으로 변환
df_result = pd.DataFrame(results)
# 컬럼 순서 정렬
df_result = df_result[['종목명', '현재가', '내일예측', '예상등락', '방향']]

print(df_result.to_string(index=False))
print("=" * 60)