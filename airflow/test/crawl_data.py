import okx.MarketData as MarketData
import pandas as pd
import ta
import os
from datetime import datetime, timedelta

def crawl_data():
    flag = "0"  # Production trading:0 , demo trading:1
    marketDataAPI = MarketData.MarketAPI(flag=flag)
    id = "BTC-USDT" # kind of coin
    bar = '4H' # kind of bar 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H
    limit = "300"  # Số lượng records tối đa cho mỗi request
    
    result = marketDataAPI.get_candlesticks(
        instId=id,
        bar=bar,
        limit=limit
    )
    
    # Convert data to DataFrame
    rows = []
    try:
        if result and 'data' in result:
            for entry in result['data']:
                row = {
                    'ts': entry[0],
                    'o': entry[1],
                    'h': entry[2],
                    'l': entry[3],
                    'c': entry[4],
                    'vol': entry[5],
                    'volCcy': entry[6],
                    'volCcyQuote': entry[7],
                    'confirm': entry[8]
                }
                rows.append(row)

        df = pd.DataFrame(rows)
        df['ts'] = pd.to_datetime(df['ts'].astype('int64'), unit='ms')
        
        # Lấy thời điểm 1 tháng trước
        one_month_ago = datetime.now() - timedelta(days=30)
        # Lọc dữ liệu trong 1 tháng
        df = df[df['ts'] >= one_month_ago]
        
        df[['o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote']] = df[['o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote']].astype('float64')
        
        print("Shape của DataFrame:", df.shape)
        print("Thời gian bắt đầu:", df['ts'].min())
        print("Thời gian kết thúc:", df['ts'].max())

        df_indi = ta.add_all_ta_features(df, "o", "h", "l", "c", "vol", fillna=True)
        df_indi.dropna(inplace=True)
        os.makedirs("raw_data", exist_ok=True)
        df_indi.to_csv("raw_data/coin_data.csv", index=False)
        return df_indi
    except Exception as e:
        print(f"Lỗi khi xử lý dữ liệu: {e}")
        return None

if __name__ == "__main__":
    print("Bắt đầu crawl dữ liệu...")
    df = crawl_data()
    if df is not None:
        print("\nCrawl dữ liệu thành công!")
        print("Columns trong DataFrame:", df.columns.tolist())
    else:
        print("\nCrawl dữ liệu thất bại!")