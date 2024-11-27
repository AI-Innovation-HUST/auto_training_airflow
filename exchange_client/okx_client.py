import okx.Trade as Trade
import okx.Account as Account

class OKXClient:
    def __init__(self, api_key: str, secret_key: str, passphrase: str, is_sandbox: bool = False):
      
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.is_sandbox = is_sandbox

        
        self.trade_client = Trade.TradeAPI(
            api_key=self.api_key,
            api_secret_key=self.secret_key,
            passphrase=self.passphrase,
            debug=True,
            flag="1" if self.is_sandbox else "0"
        )
        self.account_client = Account.AccountAPI(
            api_key=self.api_key,
            api_secret_key=self.secret_key,
            passphrase=self.passphrase,
            debug=True,
            flag="1" if self.is_sandbox else "0"
        )

    def place_order(self,
                    clOrdId: str,
                    inst_id: str,
                    side: str,                    
                    size: str,
                    posSide: str,
                    tpTriggerPx: str ,
                    slTriggerPx: str,
                   ):
        """Place an order on OKX.
        Args:
            clOrdId (str): là order id của lệnh là 1 chuỗi bất kỳ sử dụng nó để check status.
            inst_id (str): là cặp tiền tệ trade ở đây chỉ trade  Perpetual Futures (swap) ví dụ BTC-USDT-SWAP. thêm -SWAP vào sau
            side (str): Order side (buy or sell).
            size (str): Order size. size này là bội số của base unit của cặp tiền tệ đó.
                        Đê check base unit hãy sử dụng hàm get_base_unit. Ví dụ 1 base unit của BTC-USDT-SWAP là 0.01, ETH là 0.1
            posSide (str): Position side (long or short).
            tpTriggerPx (str): Take profit trigger price.
            slTriggerPx (str): Stop loss trigger price."""
        
        try:         

            order = self.trade_client.place_order(
            clOrdId=clOrdId,
            instId=inst_id,  
            tdMode="cross",       
            side=side,        
            ordType="market",      
            sz=size,               
            posSide=posSide ,        
            tpTriggerPx=tpTriggerPx,
            tpOrdPx="-1",
            slTriggerPx=slTriggerPx,
            slOrdPx="-1",
            )         

            return order
        except Exception as e:
            print(f"Error placing order: {e}")
            return None

    def get_order_status(self, order_id: str):
       
        try:
            status = self.trade_client.get_order(order_id)
            return status
        except Exception as e:
            print(f"Error fetching order status: {e}")
            return None

    def close_order(self, inst_id: str,posSide : str):
        
        try:
            result = self.trade_client.close_positions(
            instId=inst_id,
            mgnMode="cross",
            posSide= posSide
            )
            return result
        except Exception as e:
            print(f"Error cancelling order: {e}")
            return None
    def get_account_balance(self, ccy: str):
            
        try:
            balance = self.account_client.get_account_balance(ccy)
            return balance
        except Exception as e:
            print(f"Error fetching account balance: {e}")
            return None
    def set_position_mode(self):
        
        try:
            result = self.account_client.set_position_mode(posMode="long_short_mode")
            return result
        except Exception as e:
            print(f"Error setting position mode: {e}")
            return None
    def get_base_unit(self, inst_id: str):
        
        try:
            result = self.account_client.get_instruments(instType="SWAP",instId=inst_id)
            ctVal = result['data'][0]['ctVal']
            return ctVal
        except Exception as e:
            print(f"Error getting base unit: {e}")
            return None