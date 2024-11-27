import okx.MarketData as MarketData
import pandas as pd
import time
from datetime import datetime, timedelta
import os

class OKXDataFetcher:
    BAR_MINUTES = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1H': 60,
        '2H': 120,
        '4H': 240
    }

    def __init__(self):
        self.api = MarketData.MarketAPI(flag="0")

    def _calculate_reverse_time(self, bar):
        minutes = self.BAR_MINUTES.get(bar, 1)
        return 1000 * 60 * minutes * 100

    def fetch_by_days(self, instId="BTC-USDT", bar="1m", days=7):
        """Lấy dữ liệu từ hiện tại lùi về N ngày"""
        all_data = []
        current_after = int(datetime.now().timestamp() * 1000)
        target_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        reverse_time = self._calculate_reverse_time(bar)

        while True:
            current_before = current_after - reverse_time
            if current_before <= target_ts:
                current_before = target_ts

            result = self.api.get_history_candlesticks(
                instId=instId,
                bar=bar,
                after=str(current_after),
                before=str(current_before)
            )
            
            if not result or not result.get('data'):
                break
            
            all_data.extend(result['data'])
            current_after = current_before
            
            print(f"Fetched {len(all_data)} {bar} candles")
            time.sleep(0.05)
            
            if current_after <= target_ts:
                break

        return self._create_dataframe(all_data)

    def fetch_by_bars(self, instId="BTC-USDT", bar="1m", total_candles=1000):
        """Lấy N nến từ hiện tại về quá khứ"""
        all_data = []
        current_after = int(datetime.now().timestamp() * 1000)
        reverse_time = self._calculate_reverse_time(bar)

        while len(all_data) < total_candles:
            current_before = current_after - reverse_time

            result = self.api.get_history_candlesticks(
                instId=instId,
                bar=bar,
                after=str(current_after),
                before=str(current_before)
            )
            
            if not result or not result.get('data'):
                break
            
            all_data.extend(result['data'])
            current_after = current_before
            
            print(f"Fetched {len(all_data)} {bar} candles")
            time.sleep(0.05)

        return self._create_dataframe(all_data[:total_candles])

    def fetch_by_range(self, instId="BTC-USDT", bar="1m", start_date=None, end_date=None):
        """Lấy dữ liệu từ ngày A đến ngày B"""
        all_data = []
        current_after = self.date_to_timestamp(end_date)
        target_ts = self.date_to_timestamp(start_date)
        reverse_time = self._calculate_reverse_time(bar)

        while True:
            current_before = current_after - reverse_time
            if current_before <= target_ts:
                current_before = target_ts

            result = self.api.get_history_candlesticks(
                instId=instId,
                bar=bar,
                after=str(current_after),
                before=str(current_before)
            )
            
            if not result or not result.get('data'):
                break
            
            all_data.extend(result['data'])
            current_after = current_before
            
            print(f"Fetched {len(all_data)} {bar} candles")
            time.sleep(0.05)
            
            if current_after <= target_ts:
                break

        return self._create_dataframe(all_data)

    def _create_dataframe(self, data):
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close',
            'volume', 'volCcy', 'volCcyQuote', 'confirm'
        ])
        
        df['date'] = pd.Series([datetime.fromtimestamp(int(x)/1000) for x in df['timestamp']])
        df.drop('timestamp', axis=1, inplace=True)
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'volCcy', 'volCcyQuote']
        df[numeric_cols] = df[numeric_cols].astype(float)
        
        return df.sort_values('date', ascending=False)

    def date_to_timestamp(self, date_str):
        """Convert date string (YYYY-MM-DD HH:mm:ss) to millisecond timestamp"""
        dt = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
        return int(dt.timestamp() * 1000)

    def save_to_csv(self, df, filename):
        df.to_csv(filename, index=False)
        print(f"Saved to {filename}")
        
        





if __name__ == "__main__":
    
    try:
        os.mkdir('crawler_data')
    except:
        pass
    
    
    fetcher = OKXDataFetcher()
    
    
    # # 1. Lấy 7 ngày gần nhất
    # df1 = fetcher.fetch_by_days("BTC-USDT", "1H", days=7)
    # fetcher.save_to_csv(df1, "crawler_data/btc_7days.csv")

    # # 2. Lấy 1000 nến gần nhất
    # df2 = fetcher.fetch_by_bars("BTC-USDT", "15m", total_candles=1000)
    # fetcher.save_to_csv(df2, "crawler_data/btc_1000candles.csv")

    # 3. Lấy dữ liệu từ ngày A đến ngày B
    df3 = fetcher.fetch_by_range(
        "BTC-USDT", 
        "1H",
        start_date="2024-03-19 00:00:00",
        end_date="2024-04-30 00:00:00"
    )
    fetcher.save_to_csv(df3, "crawler_data/btc_range.csv")