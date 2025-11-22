# %%

import os
from pathlib import Path
os.chdir(str(Path(__file__).parent.parent.parent))
from importlib import reload
from src.data_service.api_client import *
API_KEY = APIClient().get_api_key()
service = EODHDDataService(API_KEY)
fundamental_data = FundamentalData(API_KEY)
price = PriceData()

historical_company_sp500 = service.get_sp500_historical_composition()
ticker_from_exchange = service.get_ticker_list_from_exchange(exchange_code= 'US')[['Code', 'Type']]

historical_company = (historical_company_sp500
                      .merge(ticker_from_exchange,
                             left_on=["Ticker"],
                             right_on=["Code"], how="left"))
historical_company = historical_company[(historical_company["Type"] == "Common Stock") | pd.isna(historical_company["Type"])]
historical_company = historical_company.dropna(subset=['Ticker'])
historical_company['Ticker'] = [str(ticker).replace('.', '-') + ".US" for ticker in historical_company['Ticker']]
list_ticker = historical_company['Ticker'].unique()

print(f"Number of tickers: {len(list_ticker)}")
funda = service.get_fundamental_data(list_ticker)
print("running fundamental data processing...")
general = fundamental_data.process_fundamental_data(funda,list_ticker,"general")

income_statement = fundamental_data.process_fundamental_data(funda,list_ticker,"Income_Statement")
balance_sheet = fundamental_data.process_fundamental_data(funda,list_ticker,"Balance_Sheet")
cash_flow = fundamental_data.process_fundamental_data(funda,list_ticker,"Cash_Flow")
earnings = fundamental_data.process_fundamental_data(funda,list_ticker,"Earnings")
outstanding_shares = fundamental_data.process_fundamental_data(funda,list_ticker,"outstandingShares")

print("running price data processing...")

prices = price.get_price_data_for_tickers(tickers=list_ticker)
SP500Price = price.get_price_data_for_tickers(tickers=["SPY"])
Selected_Exchange = 'US'
print("writing data to parquet files")
os.chdir('data/')
historical_company_sp500.to_csv(os.path.join('SP500_Constituents.csv'), index=False)
prices.to_parquet(os.path.join(f'{Selected_Exchange}_Finalprice.parquet'))
general.to_parquet(os.path.join(f'{Selected_Exchange}_General.parquet'))
income_statement.to_parquet(os.path.join(f'{Selected_Exchange}_Income_statement.parquet'))
balance_sheet.to_parquet(os.path.join(f'{Selected_Exchange}_Balance_sheet.parquet'))
cash_flow.to_parquet(os.path.join(f'{Selected_Exchange}_Cash_flow.parquet'))
earnings.to_parquet(os.path.join(f'{Selected_Exchange}_Earnings.parquet'))
outstanding_shares.to_parquet(os.path.join(f'{Selected_Exchange}_share.parquet'))
SP500Price.to_parquet(os.path.join("SP500Price.parquet"))


# %%
