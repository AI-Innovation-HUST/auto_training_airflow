# crawl_data
1. vào file `airflow/tools/crawl_data.py`
2. lựa chọn hàm trong class cho phù hợp để xuất dữ liệu
* lấy dữ liệu trong x ngày kể từ thời điểm hiện tại

`df1 = fetcher.fetch_by_days("BTC-USDT", "1H", days=7)` 
* lấy x dữ liệu (số nến) kể từ thời điểm hiện tại

`df2 = fetcher.fetch_by_bars("BTC-USDT", "15m", total_candles=1000)`

* lấy dữ liệu từ ngày A trở về ngày B 

` df3 = fetcher.fetch_by_range(
        "BTC-USDT", 
        "1H",
        start_date="2024-03-19 00:00:00",
        end_date="2024-04-30 00:00:00"
    )`

Note: hỗ trợ các nến 1m, 5m, 15m, 30m, 1H, 2H, 4H. Sửa trong từng hàm theo từng lựa chọn về nến.

File xuất ra sẽ trở trong folder `crawler_data`
# auto_training_airflow

1. tools auto_training and crawl data from okx platform
2. Các bước khởi chạy auto tool trên local machine
   * install package bằng lệnh: sh install_all_packages.sh
   * Khởi chạy airflow webserver bằng lệnh : airflow webserver --port 8080
   * copy các file sau vào thư mục airflow(khởi tạo bảo airflow):
     * mkdir ~/airflow/dags
     * cp dataloader_v2.py ~/airflow/dags
     * cp full_pipeline.py ~/airflow/dags
     * cp model.py ~/airflow/dags
     * cp scaler.py ~/airflow/dags
     * cp transformer.py ~/airflow/dags
   * Reload lại Airflow webserver và turn on DAG trên web
