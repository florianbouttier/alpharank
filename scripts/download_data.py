import os
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

from alpharank.data.service import APIClient, EODHDDataService, FundamentalData, PriceData

def main():
    # Ensure data directory exists
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    print("Initializing services...")
    # Initialize API Client and Services
    # Note: Ensure EODHD_API_KEY is set in your environment or config.ini
    api_client = APIClient()
    api_key = api_client.get_api_key()
    
    if not api_key:
        print("Error: API Key not found. Please set EODHD_API_KEY environment variable or check config.ini")
        return

    service = EODHDDataService(api_key)
    # Correctly instantiate FundamentalData with the client, not the key string
    fundamental_data = FundamentalData(api_client)
    price = PriceData(api_client)

    print("Fetching S&P 500 historical composition...")
    historical_company_sp500 = service.get_sp500_historical_composition()
    
    print("Fetching ticker list from US exchange...")
    ticker_from_exchange = service.get_ticker_list_from_exchange(exchange_code='US')[['Code', 'Type']]

    print("Processing tickers...")
    historical_company = (historical_company_sp500
                          .merge(ticker_from_exchange,
                                 left_on=["Ticker"],
                                 right_on=["Code"], how="left"))
    
    # Filter for Common Stock or where Type is missing
    historical_company = historical_company[(historical_company["Type"] == "Common Stock") | pd.isna(historical_company["Type"])]
    historical_company = historical_company.dropna(subset=['Ticker'])
    
    # Format tickers
    historical_company['Ticker'] = [str(ticker).replace('.', '-') + ".US" for ticker in historical_company['Ticker']]
    list_ticker = historical_company['Ticker'].unique()

    print(f"Number of tickers to process: {len(list_ticker)}")
    
    # Fetch fundamental data
    print("Downloading fundamental data (this may take a while)...")
    funda = service.get_fundamental_data(list_ticker)
    
    print("Processing fundamental data...")
    general = fundamental_data.process_fundamental_data(funda, list_ticker, "general")
    income_statement = fundamental_data.process_fundamental_data(funda, list_ticker, "Income_Statement")
    balance_sheet = fundamental_data.process_fundamental_data(funda, list_ticker, "Balance_Sheet")
    cash_flow = fundamental_data.process_fundamental_data(funda, list_ticker, "Cash_Flow")
    earnings = fundamental_data.process_fundamental_data(funda, list_ticker, "Earnings")
    outstanding_shares = fundamental_data.process_fundamental_data(funda, list_ticker, "outstandingShares")

    print("Processing price data...")
    prices = price.get_price_data_for_tickers(tickers=list_ticker)
    SP500Price = price.get_price_data_for_tickers(tickers=["SPY"])
    
    Selected_Exchange = 'US'
    print(f"Writing data to parquet files in {data_dir}...")
    
    historical_company_sp500.to_csv(data_dir / 'SP500_Constituents.csv', index=False)
    
    # Helper to save parquet safely
    def save_parquet(df, filename):
        if not df.empty:
            df.to_parquet(data_dir / filename)
        else:
            print(f"Warning: DataFrame for {filename} is empty, skipping save.")

    save_parquet(prices, f'{Selected_Exchange}_Finalprice.parquet')
    save_parquet(general, f'{Selected_Exchange}_General.parquet')
    save_parquet(income_statement, f'{Selected_Exchange}_Income_statement.parquet')
    save_parquet(balance_sheet, f'{Selected_Exchange}_Balance_sheet.parquet')
    save_parquet(cash_flow, f'{Selected_Exchange}_Cash_flow.parquet')
    save_parquet(earnings, f'{Selected_Exchange}_Earnings.parquet')
    save_parquet(outstanding_shares, f'{Selected_Exchange}_share.parquet')
    save_parquet(SP500Price, "SP500Price.parquet")
    
    print("Done!")

if __name__ == "__main__":
    main()
