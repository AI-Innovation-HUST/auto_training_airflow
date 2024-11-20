
from okx_client import OKXClient
from config import API_KEY, SECRET_KEY, PASSPHRASE

def main():
    okx_client = OKXClient(API_KEY, SECRET_KEY, PASSPHRASE,True)
    # result = okx_client.place_order(inst_id="BTC-USDT-SWAP", side="buy", size="1", posSide="long", tpTriggerPx="95000", slTriggerPx="40000")
    # print(result)
    # result = okx_client.close_order(inst_id="BTC-USDT-SWAP", posSide="short")
    # print(result)
    # result = okx_client.get_account_balance("ETH")
    # print(result["data"][0]["details"][0]["eq"])
    result = okx_client.get_base_unit("BTC-USDT-SWAP")
    print(result)

if __name__ == "__main__":
    main()


