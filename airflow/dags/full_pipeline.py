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
       # Bỏ cột ts vì không cần cho việc training
       if 'ts' in df.columns:
           df = df.drop('ts', axis=1)
           
       os.makedirs("results", exist_ok=True)
       
       # Split data
       train_val_df, test_df = train_test_split(df, test_size=test_size, shuffle=False, random_state=42)
       train_df, val_df = train_test_split(train_val_df, test_size=val_size, shuffle=False, random_state=42)

       train_dataset = CoinDataset(train_df)
       val_dataset = CoinDataset(val_df.reset_index(drop=True))
       test_dataset = CoinDataset(test_df.reset_index(drop=True))
       
       print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
       return train_dataset, val_dataset, test_dataset
   except Exception as e:
       print(f"Lỗi trong load_data: {e}")
       raise e

def evaluate(model, epoch, criterion_high, criterion_low, data_source, dev):
   model.eval()
   total_loss = 0
   total_acc_h = 0
   total_acc_l = 0
   count = 0
   
   with torch.no_grad():
       for data, targets in data_source:
           targets = targets.view(-1, 2).to(dev)
           output = model(data.to(dev))
           
           # Xử lý output
           if isinstance(output, tuple):
               high, low = output
           else:
               # Nếu model trả về tensor đơn
               high = output[:, 0].view(-1, 1)
               low = output[:, 1].view(-1, 1)
           
           # Tính loss
           loss_high = criterion_high(high.float(), targets[:, 0].float())
           loss_low = criterion_low(low.float(), targets[:, 1].float())
           total_loss += (loss_high.item() + loss_low.item()) / 2
           
           # Tính accuracy
           score_h, score_l = compute_acc(high, low, targets)
           total_acc_h += score_h.item() if isinstance(score_h, torch.Tensor) else score_h
           total_acc_l += score_l.item() if isinstance(score_l, torch.Tensor) else score_l
           count += 1
   
   avg_loss = total_loss / count if count > 0 else float('inf')
   avg_acc_h = total_acc_h / count if count > 0 else 0
   avg_acc_l = total_acc_l / count if count > 0 else 0
   
   print(f"Epoch {epoch} - Validation: Loss = {avg_loss:.4f}, High Acc = {avg_acc_h:.4f}, Low Acc = {avg_acc_l:.4f}")
   
   return avg_loss, avg_acc_h, avg_acc_l

def training_transformer():
   try:
       print("Bắt đầu training...")
       df = pd.read_csv("raw_data/scaled_data.csv")
       if df.empty:
           print("Không có dữ liệu để train")
           return None

       dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
       print(f"Using device: {dev}")

       # Load và split data
       train_dataset, val_dataset, test_dataset = load_data(df)
       
       # Print một dòng dữ liệu để kiểm tra
       print("\nMột dòng dữ liệu mẫu:")
       sample_row = df.iloc[0]
       print(sample_row)
       
       train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
       val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)
       test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
       
       print("Train dataset size:", len(train_dataset))
       print("Validation dataset size:", len(val_dataset))
       print("Test dataset size:", len(test_dataset))
       
       # Check input size from data
       vocab_size = df.shape[1] - 1  # trừ đi cột ts
       print(f"Vocabulary size from data: {vocab_size}")

       # Initialize model với vocab_size từ data
       model = Transformer(n_blocks=4, d_model=16, n_heads=8, d_ff=256, dropout=0.5, vocab_size=vocab_size)
       model.to(dev)

       criterion_high = nn.MSELoss()
       criterion_low = nn.MSELoss()
       lr = 0.0001
       optimizer = torch.optim.Adam(model.parameters(), lr=lr)
       scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
       epochs = 200
       best_val_loss = float('inf')

       for epoch in range(epochs):
           model.train()
           count = 0
           cum_loss_h = 0
           cum_loss_l = 0
           accs_h = 0
           accs_l = 0

           for batch in train_loader:
               data, targets = batch
               targets = targets.view(-1,2).to(dev)
               output = model(data.to(dev))
               
               # Xử lý output của model
               if isinstance(output, tuple):
                   high, low = output
               else:
                   high = output[:, 0].view(-1, 1)
                   low = output[:, 1].view(-1, 1)
               
               # Đảm bảo kích thước đúng
               high = high.view(-1, 1)
               low = low.view(-1, 1)
               targets = targets.view(-1, 2)
               
               loss_high = criterion_high(high.float(), targets[:,0].float().view(-1, 1))
               loss_low = criterion_low(low.float(), targets[:,1].float().view(-1, 1))
               total_loss = 0.5 * loss_high + 0.5 * loss_low
               
               score_h, score_l = compute_acc(high, low, targets)
               accs_h += score_h.item() if isinstance(score_h, torch.Tensor) else score_h
               accs_l += score_l.item() if isinstance(score_l, torch.Tensor) else score_l
               cum_loss_h += loss_high.item()
               cum_loss_l += loss_low.item()
               
               total_loss.backward()
               optimizer.step()
               model.zero_grad()
               optimizer.zero_grad()
               count += 1

           print(f"Epoch {epoch}:")
           print(f"Loss high: {cum_loss_h/count:.4f}")
           print(f"Loss low: {cum_loss_l/count:.4f}")
           print(f"ACC high train: {accs_h/count:.4f}")
           print(f"ACC low train: {accs_l/count:.4f}")

           print("------START EVAL------")
           eval_loss, acc_h, acc_l = evaluate(model, epoch, criterion_high, criterion_low, val_loader, dev=dev)
           
           if eval_loss < best_val_loss:
               best_val_loss = eval_loss
               torch.save({
                   'epoch': epoch,
                   'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'loss': best_val_loss,
               }, "results/evalModel_best.pth")
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
    


