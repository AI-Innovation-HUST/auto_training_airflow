from dotenv import load_dotenv
import okx.Account as Account
import os
import okx.Trade as Trade

# Load environment variables from .env file
load_dotenv()

# Get environment variables
api_key = os.getenv("API_KEY")
secret_key = os.getenv("SECRET_KEY")    
passphrase = os.getenv("PASSPHRASE")

flag = "1"  # live trading: 0, demo trading: 1

accountAPI = Account.AccountAPI(api_key, secret_key, passphrase, False, flag)
tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)

# Set position mode
mode = accountAPI.set_position_mode(
    posMode="long_short_mode"
)

# Set leverage for MARGIN instruments under isolated-margin trade mode at pairs level.
lever = accountAPI.set_leverage(
    instId="BTC-USDT-SWAP",
    lever="100",
    mgnMode="cross"
)

# Spot mode, limit order
trade = tradeAPI.place_order(
    instId="BTC-USDT-SWAP",
    tdMode="cross",
    clOrdId="b5",
    side="buy",
    ordType="market",
    posSide='long',
    sz="1",
    tpTriggerPx=100000,
    tpOrdPx="-1",
    slTriggerPx=40000,
    slOrdPx="-1",
)

# Get account balance
result = accountAPI.get_account_balance()
print(result['data'][0]['details'][2]['availBal'])