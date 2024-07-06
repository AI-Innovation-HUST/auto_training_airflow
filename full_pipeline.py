import datetime
from datetime import timedelta 
import csv
import json
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
import time
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
def load_data(df, test_size=0.1, val_size=0.2):
    os.makedirs("results",exist_ok=True)
    scaler = ScalerData(method='minmax') # test method
    df = scaler.fit_transform(df)
    save_scaler_file = "results/scaler.pkl"
    pickle.dump(scaler, open(save_scaler_file, 'wb'))
    print("Save scaler model successfully")
    train_val_df, test_df = train_test_split(df, test_size=test_size, shuffle=False, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, shuffle=False, random_state=42)

    train_dataset = CoinDataset(train_df)
    val_dataset = CoinDataset(val_df.reset_index(drop=True))
    test_dataset = CoinDataset(test_df.reset_index(drop=True))
    return train_dataset, val_dataset, test_dataset
# Convert datetime to miliseconds
def training_transformer():
    df = pd.read_csv("raw_data/coin_data.csv")
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    train_dataset, val_dataset, test_dataset = load_data(df)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))
    print("Test dataset size:", len(test_dataset))
    model = Transformer(n_blocks=4,d_model=16,n_heads=8,d_ff=256,dropout=0.5)
    model.to(dev)
    criterion_high = nn.MSELoss()
    criterion_low = nn.MSELoss()
    lr = 0.0001 # learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    epochs = 100
    val_loss = 0
    for epoch in range(epochs):
        count = 0
        cum_loss_h = 0
        cum_loss_l = 0
        accs_h = 0
        accs_l = 0
        train_l_high = []
        train_l_low = []
        for batch in train_loader:
            data, targets = batch
            targets = targets.view(-1,2).to(dev)
            high,low = model(data.to(dev))
            high = high.view(-1,1)
            low = low.view(-1,1)
            # print(output.shape)
            loss_high = criterion_high(high.float(),targets[:,0].float())
            loss_low = criterion_low(low.float(),targets[:,1].float)
            total_loss = 0.5 * loss_high + 0.5 * loss_low
            train_l_high.append(loss_high.item())
            train_l_low.append(loss_low.item())
            score_h,score_l = compute_acc(high,low,targets)
            accs_h += score_h
            accs_l += score_l
            cum_loss_h += loss_high
            cum_loss_l += loss_low
            total_loss.backward()
            optimizer.step()
            model.zero_grad()
            optimizer.zero_grad()
            count+=1
        print(epoch,"Loss high: ",(cum_loss_h/count).item(),"Loss_low:",(cum_loss_l/count).item(),"Total loss: ",(total_loss/count).item(),"ACC high train:",(accs_h/count).item(),"ACC low train:",(accs_l/count).item())
        print("------START EVAL------")
        eval_loss,acc_h,acc_l = evaluate(model,epoch,criterion_high,criterion_low,val_loader,dev=dev)
        val_loss = eval_loss
        print(epoch,"Loss: ",(cum_loss_h/count).item()," Valid_loss: ",eval_loss,"Valid high acc: ",acc_h,"valid low acc",acc_l)
        if eval_loss < val_loss[-1]:
            torch.save(model,"results/evalModel_best.pth")
    
def datetime_to_ms(dt):
    dt = datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S.%f')
    return int(dt.timestamp() * 1000)

# Convert miliseconds to datetime
def ms_to_datetime(ms):
    return datetime.datetime.fromtimestamp(ms / 1000.0)

print(ms_to_datetime(1640969940000))



def convert_bar_to_timestamp(bar):
    bar_to_timestamp = {
        '1m': 1,
        '3m': 3,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1H': 60,
        '2H': 120,
        '4H': 240
    }
    
    return bar_to_timestamp.get(bar, "Invalid bar")

# Retrieve history candlestick charts from recent years
def get_candlesticks(marketDataAPI,instId, start_ms, end_ms, bar):
    try:
        result = marketDataAPI.get_history_candlesticks(
            instId=instId,
            after=(start_ms),
            before=(end_ms),
            bar=bar,
        )
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None
    

def crawl_data():
    flag = "0"  # Production trading:0 , demo trading:1
    marketDataAPI =  MarketData.MarketAPI(flag=flag)
    current_time = datetime.datetime.now()
    id = "BTC-USDT" # kind of coin
    bar = '4H' # kind of bar 1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H
    print("Thời điểm hiện tại:", current_time)
    one_week = timedelta(days=7)  
    # Thời điểm một tuần sau hiện tại  
    one_week_later = current_time + one_week
    timestamp = convert_bar_to_timestamp(bar)
    before = datetime_to_ms(current_time)
    after = datetime_to_ms(one_week_later)
    end = after
    result_all = []
    # Crawl data from okx
    while after - before >= 5940000*timestamp:
        result = get_candlesticks(id, after, before, bar)
        result_all.append(result)
        after = after - 5940000*timestamp - 60000*timestamp

    result = get_candlesticks(marketDataAPI,id, after, before, bar)
    result_all.append(result)
    # Convert data to DataFrame
    rows = []
    try:
        for item in result_all:
            for entry in item['data']:
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
        df[['o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote']] = df[['o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote']].astype('float64')
    except Exception as e:
        print(f"Error: {e}")
    df_indi = ta.add_all_ta_features(df, "o", "h", "l", "c", "vol", fillna=True)
    df_indi.dropna(inplace=True)
    os.makedirs("raw_data",exist_ok=True)
    df_indi.to_csv("raw_data/coin_data.csv",index=False)
    return df_indi

def process_data(**kwargs):
    from sklearn.preprocessing import MinMaxScaler
    ti = kwargs['ti']  
    crawled_data = crawl_data()
    scaler = MinMaxScaler(feature_range=(0,1)) # test method
    df = scaler.fit_transform(crawled_data)
    save_scaler_file = "scaler.pkl"
    pickle.dump(scaler, open(save_scaler_file, 'wb'))
    print("Save scaler model successfully")
    return df
    
    
default_args = {  
    'owner': 'airflow',  
    'depends_on_past': False,  
    'start_date': datetime.datetime.now(),  
    'email_on_failure': False,  
    'email_on_retry': False,  
    'retries': 1  
}  
  
dag = DAG(  
    'crawl_data_okx',  
    default_args=default_args,  
    description='Crawl data and store as DataFrame',  
    schedule_interval='@monthly',  
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
    task_id = 'training model',
    python_callable = training_transformer,
    provide_context = True,
    dag =dag
    
)
crawl_data_task >> process_data_task >> training_model
    

    


