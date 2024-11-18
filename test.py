
# import okx.MarketData as MarketData
# import pandas as pd
# import time
# from datetime import datetime, timedelta

# api = MarketData.MarketAPI(flag="0")
# all_data = []

# # Timestamp hiện tại
# current_after = int(datetime.now().replace(second=0, microsecond=0).timestamp() * 1000)
# target_ts = int((datetime.now() - timedelta(days=2)).timestamp() * 1000)

# while True:
#     current_before = current_after - 1000*60*100  # Lấy 300 nến 1m về quá khứ
    
#     if current_before <= target_ts:
#         current_before = target_ts

#     result = api.get_history_candlesticks(
#         instId="BTC-USDT", 
#         bar="1m",
#         after=str(current_after),
#         before=str(current_before),
#     )
    
#     if not result or not result.get('data'):
#         break
    
#     candles = result['data']
#     all_data.extend(candles)
    
#     # Update timestamps cho lần sau
#     current_after = current_before
    
#     print(f"Fetched {len(all_data)} candles")
#     time.sleep(0.05)
    
#     if current_after <= target_ts:
#         break

# df = pd.DataFrame(all_data, columns=[
#     'timestamp', 'open', 'high', 'low', 'close',
#     'volume', 'volCcy', 'volCcyQuote', 'confirm'
# ])

# df['date'] = pd.Series([datetime.fromtimestamp(int(x)/1000) for x in df['timestamp']])
# df.drop('timestamp', axis=1, inplace=True)
# df[['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote']] = df[['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote']].astype(float)

# df = df.sort_values('date', ascending=False)
# df.to_csv("btc_data.csv", index=False)



import time
from datetime import datetime
from datetime import datetime, timedelta

# Lấy timestamp hiện tại (milliseconds)
from datetime import datetime
timestamp_ms = int(datetime.now().replace(second=0, microsecond=0).timestamp() * 1000)
current_after = int(datetime.now().replace(second=0, microsecond=0).timestamp() * 1000)
target_ts = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
# # Chuyển từ timestamp (ms) sang datetime
# timestamp_ms = 1731952080000
# dt = datetime.fromtimestamp(timestamp_ms / 1000)
# print(timestamp_ms)
# print(dt)  # VD: 2023-11-20 15:22:24.123000


import okx.MarketData as MarketData

flag = "0"  # Production trading:0 , demo trading:1

marketDataAPI =  MarketData.MarketAPI(flag=flag)

# Retrieve the candlestick charts
result = marketDataAPI.get_candlesticks(
    instId="BTC-USDT",
    bar="1m",
    after=timestamp_ms,
    # before="1731941760000"
)
print(result)

dt = datetime.fromtimestamp(timestamp_ms / 1000)
print(dt)  # VD: 2023-11-20 15:22:24.123000