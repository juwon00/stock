import numpy as np
import requests
import json
import talib
import yaml
import FinanceDataReader as fdr
from matplotlib import pyplot as plt

with open('config.yaml', encoding='UTF-8') as f:
    _cfg = yaml.load(f, Loader=yaml.FullLoader)
APP_KEY = _cfg['APP_KEY']
APP_SECRET = _cfg['APP_SECRET']
ACCESS_TOKEN = ""
CANO = _cfg['CANO']
ACNT_PRDT_CD = _cfg['ACNT_PRDT_CD']
DISCORD_WEBHOOK_URL = _cfg['DISCORD_WEBHOOK_URL']
URL_BASE = _cfg['URL_BASE']


def get_access_token():
    """토큰 발급"""
    headers = {"content-type": "application/json"}
    body = {
        "grant_type": "client_credentials",
        "appkey": APP_KEY,
        "appsecret": APP_SECRET
    }
    PATH = "oauth2/tokenP"
    URL = f"{URL_BASE}/{PATH}"
    res = requests.post(URL, headers=headers, data=json.dumps(body))
    print(json.dumps(res.json(), ensure_ascii=False, indent=3))
    ACCESS_TOKEN = res.json()["access_token"]
    return ACCESS_TOKEN


def get_current_price(market="NAS", code="AAPL"):
    """현재 체결가"""
    PATH = "uapi/overseas-price/v1/quotations/price"
    URL = f"{URL_BASE}/{PATH}"
    headers = {"Content-Type": "application/json",
               "authorization": f"Bearer {ACCESS_TOKEN}",
               "appKey": APP_KEY,
               "appSecret": APP_SECRET,
               "tr_id": "HHDFS00000300"}
    params = {
        "AUTH": "",
        "EXCD": market,
        "SYMB": code,
    }
    res = requests.get(URL, headers=headers, params=params)
    # print(json.dumps(res.json(), ensure_ascii=False, indent=3))
    return res.json()


def get_period_price(market="NAS", code="QQQ"):
    """기간별 시세"""
    PATH = "/uapi/overseas-price/v1/quotations/dailyprice"
    URL = f"{URL_BASE}/{PATH}"
    headers = {"Content-Type": "application/json",
               "authorization": f"Bearer {ACCESS_TOKEN}",
               "appKey": APP_KEY,
               "appSecret": APP_SECRET,
               "tr_id": "HHDFS76240000"}
    params = {
        "AUTH": "",
        "EXCD": market,
        "SYMB": code,
        "GUBN": 0,  # 0-일, 1-주, 2-월
        "BYMD": "",
        "MODP": 1,
    }
    res = requests.get(URL, headers=headers, params=params)
    # print(json.dumps(res.json(), ensure_ascii=False, indent=3))
    return res.json()


def get_minute_price(market, code):
    """분봉 조회"""
    PATH = "/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"
    URL = f"{URL_BASE}/{PATH}"
    headers = {"Content-Type": "application/json",
               "authorization": f"Bearer {ACCESS_TOKEN}",
               "appKey": APP_KEY,
               "appSecret": APP_SECRET,
               "tr_id": "HHDFS76950200"}
    params = {
        "AUTH": "",
        "EXCD": market,
        "SYMB": code,
        "NMIN": 3,  # 분 단위
        "PINC": 1,  # 0:당일 1:전일포함
        "NEXT": "",
        "NREC": 120,
        "FILL": "",
        "KEYB": "",
    }
    res = requests.get(URL, headers=headers, params=params)
    # print(json.dumps(res.json(), ensure_ascii=False, indent=3))
    return res.json()


def draw_day_graph():
    period_price = get_period_price()['output2']  # 일봉 조회

    close = np.array([float(p['clos']) for p in period_price])[::-1]
    high = np.array([float(p['high']) for p in period_price])[::-1]
    low = np.array([float(p['low']) for p in period_price])[::-1]

    macd, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    adx = talib.ADX(high=high, low=low, close=close, timeperiod=14)
    rsi = talib.RSI(close, timeperiod=14)
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)

    # 화면 나누기
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 6), sharex=True)

    # 첫 번째 화면에 가격 데이터 그리기
    ax1.plot(close, label='Price')
    ax1.legend()
    ax1.set_title('Price')

    # 두 번째 화면에 MACD 그래프 그리기
    ax2.plot(macd, label='MACD')
    ax2.plot(signal, label='Signal')
    ax2.axhline(y=0, color='red', linestyle='--', label='x=0')  # x=0에서 빨간색 점선 추가
    ax2.legend()
    ax2.set_title('MACD')

    # 세 번째 화면에 ADX 그래프 그리기
    ax3.plot(adx, label='ADX')
    ax3.legend()
    ax3.set_title('ADX')

    # 네 번째 화면에 RSI 그래프 그리기
    ax4.plot(rsi, label='RSI')
    ax4.legend()
    ax4.set_title('RSI')

    # 다섯 번째 화면에 Stochastic 그래프 그리기
    ax5.plot(slowk, label='SlowK')
    ax5.plot(slowd, label='SlowD')
    ax5.legend()
    ax5.set_title('Stoch')

    # x축 레이블은 맨 아래의 그래프에만 나타나도록 함
    # plt.xlabel('Index')

    # 그래프 표시
    plt.tight_layout()  # 각각의 그래프 간 간격을 조정
    plt.show()


# print(get_access_token())

# nasdaq = fdr.StockListing('NASDAQ')
# # nasdaq['Indexes'] = 'NASDAQ'
# print('nasdaq :', nasdaq.shape)
# print(nasdaq)


# s&p500 불러오고 개별적으로 계산
snp500 = fdr.StockListing('S&P500')
for s in snp500['Symbol']:
    print(s)
    period_price = get_minute_price('NYS', s)['output2']  # 분봉 조회

    close = np.array([float(p['last']) for p in period_price])[::-1]
    high = np.array([float(p['high']) for p in period_price])[::-1]
    low = np.array([float(p['low']) for p in period_price])[::-1]
    time = np.array([float(p['xhms']) for p in period_price])[::-1]

    macd, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    adx = talib.ADX(high=high, low=low, close=close, timeperiod=14)
    rsi = talib.RSI(close, timeperiod=14)
    slowk, slowd = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowd_period=3)

    buy = False
    for i in range(len(macd)):
        # print(close[i], "    ", macd[i], "   ", signal[i], "    ", adx[i], "    ", i)
        if macd[i - 2] == "nan":
            continue
        if macd[i - 2] > macd[i - 1] and macd[i - 1] < macd[i] and adx[i - 2] < adx[i - 1] and adx[i - 1] > adx[i] \
                and adx[i - 1] > 30 and rsi[i - 1] < 30 and slowk[i - 1] < 25:
            print(time[i], i, "find", close[i], "   ", macd[i - 1], "   ", adx[i - 1])
            buy = True

        if buy and macd[i-1] > signal[i-1] and macd[i] <= signal[i]:
            print(time[i], i, "sell", close[i], "   ", macd[i - 1], "   ", adx[i - 1])
            buy = False
