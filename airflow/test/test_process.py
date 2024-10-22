import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

def process_data():
    try:
        print("Đọc dữ liệu từ file coin_data.csv")
        crawled_data = pd.read_csv("raw_data/coin_data.csv")
        
        if crawled_data is not None:
            # Chỉ chọn các cột số để scale
            numeric_columns = crawled_data.select_dtypes(include=['float64', 'int64']).columns
            print("Các cột số sẽ được scale:", numeric_columns.tolist())
            
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
            print("Đã lưu dữ liệu đã scale vào scaled_data.csv")
            return scaled_data
            
    except Exception as e:
        print(f"Lỗi trong quá trình scale dữ liệu: {e}")
        return None

if __name__ == "__main__":
    print("Bắt đầu xử lý dữ liệu...")
    df = process_data()
    if df is not None:
        print("\nXử lý dữ liệu thành công!")
    else:
        print("\nXử lý dữ liệu thất bại!")