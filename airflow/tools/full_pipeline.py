import torch
import pandas as pd
import ta
from airflow import DAG  
from airflow.operators.python_operator import PythonOperator
import pickle
import okx.MarketData as MarketData
import os
import datetime
import torch
import torch.nn as nn
from model import *
from dataloader_v2 import *
from sklearn.model_selection import train_test_split
import ta
from scaler import *
from datetime import datetime, timedelta
import os


def load_data(df, test_size=0.1, val_size=0.2):
    try:
        os.makedirs("results", exist_ok=True)
        scaler = ScalerData(method='minmax')
        df = scaler.fit_transform(df)
        
        # Lưu scaler với context manager
        save_scaler_file = "results/scaler.pkl"
        with open(save_scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print("Save scaler model successfully")
        
        # Split data
        train_val_df, test_df = train_test_split(df, test_size=test_size, shuffle=False, random_state=42)
        train_df, val_df = train_test_split(train_val_df, test_size=val_size, shuffle=False, random_state=42)

        # Create datasets
        train_dataset = CoinDataset(train_df)
        val_dataset = CoinDataset(val_df.reset_index(drop=True))
        test_dataset = CoinDataset(test_df.reset_index(drop=True))
        
        print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
        return train_dataset, val_dataset, test_dataset
    except Exception as e:
        print(f"Lỗi trong load_data: {e}")
        raise e

def training_transformer(**kwargs):
    try:
        # Đọc dữ liệu đã được xử lý
        df = pd.read_csv("raw_data/coin_data.csv")
        if df.empty:
            print("Không có dữ liệu để train")
            return None

        # Setup device
        dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {dev}")

        # Load và split data
        train_dataset, val_dataset, test_dataset = load_data(df)
        
        # Setup dataloaders
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
        
        print("Train dataset size:", len(train_dataset))
        print("Validation dataset size:", len(val_dataset))
        print("Test dataset size:", len(test_dataset))

        # Initialize model
        model = Transformer(n_blocks=4, d_model=16, n_heads=8, d_ff=256, dropout=0.5)
        model.to(dev)

        # Setup training
        criterion_high = nn.MSELoss()
        criterion_low = nn.MSELoss()
        lr = 0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        epochs = 100
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            count = 0
            cum_loss_h = 0
            cum_loss_l = 0
            accs_h = 0
            accs_l = 0

            # Training step
            for batch in train_loader:
                data, targets = batch
                targets = targets.view(-1,2).to(dev)
                high, low = model(data.to(dev))
                high = high.view(-1,1)
                low = low.view(-1,1)
                
                loss_high = criterion_high(high.float(), targets[:,0].float())
                loss_low = criterion_low(low.float(), targets[:,1].float())
                total_loss = 0.5 * loss_high + 0.5 * loss_low
                
                score_h, score_l = compute_acc(high, low, targets)
                accs_h += score_h
                accs_l += score_l
                cum_loss_h += loss_high
                cum_loss_l += loss_low
                
                total_loss.backward()
                optimizer.step()
                model.zero_grad()
                optimizer.zero_grad()
                count += 1

            # Print training metrics
            print(f"Epoch {epoch}:")
            print(f"Loss high: {(cum_loss_h/count).item():.4f}")
            print(f"Loss low: {(cum_loss_l/count).item():.4f}")
            print(f"ACC high train: {(accs_h/count).item():.4f}")
            print(f"ACC low train: {(accs_l/count).item():.4f}")

            # Validation step
            print("------START EVAL------")
            eval_loss, acc_h, acc_l = evaluate(model, epoch, criterion_high, criterion_low, val_loader, dev=dev)
            print(f"Valid_loss: {eval_loss:.4f}, Valid high acc: {acc_h:.4f}, valid low acc: {acc_l:.4f}")

            # Save best model
            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                torch.save(model.state_dict(), "results/evalModel_best.pth")
                print(f"Saved new best model with validation loss: {best_val_loss:.4f}")

            scheduler.step()

        return best_val_loss
        
    except Exception as e:
        print(f"Lỗi trong training_transformer: {e}")
        raise e
    
    
        
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

def process_data(**kwargs):
    from sklearn.preprocessing import MinMaxScaler
    ti = kwargs['ti']  
    crawled_data = crawl_data()
    
    if crawled_data is not None:
        try:
            # Chỉ chọn các cột số để scale
            numeric_columns = crawled_data.select_dtypes(include=['float64', 'int64']).columns
            scaler = MinMaxScaler(feature_range=(0,1))
            
            # Scale dữ liệu
            scaled_data = crawled_data.copy()
            scaled_data[numeric_columns] = scaler.fit_transform(crawled_data[numeric_columns])
            
            # Lưu scaler
            os.makedirs("model", exist_ok=True)
            save_scaler_file = "model/scaler.pkl"
            with open(save_scaler_file, 'wb') as f:
                pickle.dump(scaler, f)
            print("Save scaler model successfully")
            
            # Lưu dữ liệu đã scale
            scaled_data.to_csv("raw_data/scaled_data.csv", index=False)
            print("Shape của dữ liệu đã scale:", scaled_data.shape)
            return scaled_data
            
        except Exception as e:
            print(f"Lỗi trong quá trình scale dữ liệu: {e}")
            return None
    else:
        print("Không có dữ liệu để xử lý")
        return None
    
    
default_args = {  
    'owner': 'airflow',  
    'depends_on_past': False,  
    'start_date': datetime.now(),  
    'email_on_failure': False,  
    'email_on_retry': False,  
    'retries': 1  
}  
  
dag = DAG(  
    'auto_training_crawling_data_okx',  
    default_args=default_args,  
    description='Crawl data and store as DataFrame',  
    schedule_interval='@weekly',  
    catchup=False  
)  
  
crawl_data_task = PythonOperator(  
    task_id='crawl_data_task',  
    python_callable=crawl_data,  
    provide_context=True,  
    dag=dag  
)  
  
process_data_task = PythonOperator(  
    task_id='process_data_task',  
    python_callable=process_data,  
    provide_context=True,  
    dag=dag  
)  

training_model = PythonOperator(
    task_id='training_model',
    python_callable=training_transformer,
    provide_context=True,
    dag=dag
)

crawl_data_task >> process_data_task >> training_model
    


