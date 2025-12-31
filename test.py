import yfinance as yf
import matplotlib.pyplot as plt

# 1. 삼성전자(005930.KS) 10년치 데이터 가져오기
print("데이터 다운로드 중...")
samsung = yf.Ticker("005930.KS")
df = samsung.history(period="40y")

# 2. 데이터가 잘 왔는지 터미널에 살짝 보여주기
print("\n[데이터 확인]")
print(df.tail()) # 끝부분 5줄만 출력

# 3. 그래프 그려서 파일로 저장하기
print("\n그래프 그리는 중...")
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Close'], label='Samsung Electronics')
plt.title('Samsung Stock Price (40 Years)')
plt.xlabel('Date')
plt.ylabel('Price (KRW)')
plt.legend()
plt.grid(True)

# 화면에 띄우는 대신 파일로 저장
plt.savefig('samsung_chart.png')
print("\n완료! 왼쪽 파일 목록에 'samsung_chart.png'가 생겼습니다.")