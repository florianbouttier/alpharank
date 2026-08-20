import os
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent

from alpharank.data.lineage import create_snapshot
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
    written_files = {}

    constituents_path = data_dir / 'SP500_Constituents.csv'
    historical_company_sp500.to_csv(constituents_path, index=False)
    written_files["sp500_constituents"] = constituents_path
    
    # Helper to save parquet safely
    def save_parquet(df, filename):
        if not df.empty:
            path = data_dir / filename
            df.to_parquet(path)
            written_files[filename.replace(".parquet", "").lower()] = path
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
    manifest = create_snapshot(
        data_dir=data_dir,
        files=written_files,
        frames={
            "sp500_constituents": historical_company_sp500,
            f"{Selected_Exchange}_finalprice".lower(): prices,
            f"{Selected_Exchange}_general".lower(): general,
            f"{Selected_Exchange}_income_statement".lower(): income_statement,
            f"{Selected_Exchange}_balance_sheet".lower(): balance_sheet,
            f"{Selected_Exchange}_cash_flow".lower(): cash_flow,
            f"{Selected_Exchange}_earnings".lower(): earnings,
            f"{Selected_Exchange}_share".lower(): outstanding_shares,
            "sp500price": SP500Price,
        },
    )
    print(f"Snapshot created: {manifest['snapshot_dir']}")
    print(f"Latest manifest: {data_dir / 'latest_snapshot.json'}")

    print("Done!")

if __name__ == "__main__":
    main()
