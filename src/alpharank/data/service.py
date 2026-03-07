import os
import requests
import pandas as pd
import configparser
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from io import StringIO


class APIClient:
    """Wrapper POO pour EODHD API"""
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API client
        
        Args:
            api_key: EODHD API key (optional, will use environment/config if None)
        """
        self.api_key = api_key if api_key else self.get_api_key()
        self.base_url = "https://eodhd.com/api"
        self.fmt = "json"

    def get_api_key(self) -> str:
        """
        Get API key from environment file or config file
        
        Returns:
            str: EODHD API key
        """
        # First try to get from .env file
        load_dotenv()
        api_key = os.getenv('EODHD_API_KEY')
        
        # If not found, try from config.ini
        if not api_key:
            env_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) if '__file__' in globals() else os.getcwd()
            config = configparser.ConfigParser()
            config.read(os.path.join(env_dir, 'config.ini'))
            api_key = config['API_KEYS']['MY_API_KEY']
        
        return api_key

    def get(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Any:
        """
        Perform a GET request to the API
        
        Args:
            endpoint: API endpoint
            params: Additional request parameters
            
        Returns:
            Data returned by the API
            
        Raises:
            Exception: If API returns error status code
        """
        url = f"{self.base_url}/{endpoint}"
        request_params = {"api_token": self.api_key, "fmt": self.fmt}
        
        if params:
            request_params.update(params)
            
        response = requests.get(url, params=request_params)
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
            
        return response.json()


class ExchangeData:
    """Class for managing exchange-related data"""
    
    def __init__(self, client: Optional[APIClient] = None):
        """
        Initialize exchange data manager
        
        Args:
            client: EODHD API client (optional, will create a new one if None)
        """
        self.client = client if client else APIClient()
    
    def get_supported_exchanges(self) -> pd.DataFrame:
        """
        Retrieve list of supported exchanges
        
        Returns:
            DataFrame containing supported exchanges
        """
        data = self.client.get(endpoint="exchanges-list/")
        return pd.DataFrame(data)
    
    def get_tickers_from_exchange(self, exchange_code: str) -> pd.DataFrame:
        """
        Download all symbols from a specific exchange
        
        Args:
            exchange_code: Exchange code (e.g., 'US', 'LSE')
            
        Returns:
            DataFrame containing exchange symbols
        """
        endpoint = f"exchange-symbol-list/{exchange_code}"
        data = self.client.get(endpoint)
        return pd.DataFrame(data)


class IndexData:
    """Class for managing index-related data"""
    
    def __init__(self, client: Optional[APIClient] = None):
        """
        Initialize index data manager
        
        Args:
            client: EODHD API client (optional, will create a new one if None)
        """
        self.client = client if client else APIClient()
    
    def get_sp500_components(self) -> pd.DataFrame:
        """
        Retrieve current S&P 500 components
        
        Returns:
            DataFrame containing S&P 500 components
        """
        endpoint = "fundamentals/GSPC.INDX"
        data = self.client.get(endpoint)
        data = list(data.items())[2]
        return pd.DataFrame.from_dict(data[1], orient='index')
    
    def get_historical_sp500(self) -> pd.DataFrame:
        """
        Retrieve S&P 500 component history since 1990
        
        Returns:
            DataFrame containing S&P 500 component history
        """
        # This method uses Wikipedia data
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract current constituents
        current_constituents_table = soup.find('table', {'id': 'constituents'})
        if current_constituents_table is None:
            raise ValueError("No table with id='constituents' found on Wikipedia page.")
        current_constituents = pd.read_html(StringIO(str(current_constituents_table)))[0]
        
        # Extract historical changes
        changes_table = soup.find('table', {'id': 'changes'})
        if changes_table is None:
            raise ValueError("No table with id='changes' found on Wikipedia page.")

        changes = pd.read_html(StringIO(str(changes_table)), header=0)[0]
        if changes.shape[1] < 6:
            raise ValueError(f"Unexpected 'changes' table format (columns={changes.shape[1]}).")

        # Keep the expected 6 fields and normalize names regardless of source header labels.
        changes = changes.iloc[:, :6].copy()
        changes.columns = ['Date', 'AddTicker', 'AddName', 'RemovedTicker', 'RemovedName', 'Reason']

        # Wikipedia includes repeated header rows in tbody; remove them generically.
        changes = changes[changes['Date'].astype(str).str.lower() != 'effective date'].reset_index(drop=True)
        changes['Date'] = pd.to_datetime(changes['Date'], format='mixed', errors='coerce')
        changes['year'] = changes['Date'].dt.year
        changes['month'] = changes['Date'].dt.month
        
        # Build history
        current_month = pd.to_datetime(datetime.now().strftime('%Y-%m-01'))
        month_seq = pd.date_range(start='1990-01-01', end=current_month, freq='MS')[::-1]
        spx_stocks = current_constituents.assign(Date=current_month)[['Date', 'Symbol', 'Security']]
        spx_stocks.columns = ['Date', 'Ticker', 'Name']
        last_run_stocks = spx_stocks.copy()
        
        for d in month_seq[1:]:
            y, m = d.year, d.month
            changes_in_month = changes[(changes['year'] == y) & (changes['month'] == m)]
            
            tickers_to_keep = last_run_stocks[~last_run_stocks['Ticker'].isin(changes_in_month['AddTicker'])].assign(Date=d)
            tickers_to_add = changes_in_month[changes_in_month['RemovedTicker'] != ''][['RemovedTicker', 'RemovedName']].assign(Date=d)
            tickers_to_add.columns = ['Ticker', 'Name', 'Date']
            
            this_month = pd.concat([tickers_to_keep, tickers_to_add])
            spx_stocks = pd.concat([spx_stocks, this_month])
            
            last_run_stocks = this_month
        
        return spx_stocks
    
    def get_sp500_data(self) -> pd.DataFrame:
        """
        Download S&P 500 ETF (SPY) data as a proxy for S&P 500
        
        Returns:
            DataFrame containing S&P 500 data
        """
        ticker = "SPY"
        
        # Get technical and price data
        ticker_data = PriceData(self.client)
        price_data = ticker_data.get_raw_price_data(ticker)
        
        fundamental_data = FundamentalData(self.client)
        technical_data = fundamental_data.get_technical_data(ticker)
        
        if not price_data.empty:
            # Merge data
            combined_data = technical_data.merge(
                price_data[['date', 'adjusted_close']], 
                on='date', 
                how='left'
            )
            combined_data['ticker'] = ticker
            return combined_data
        
        return pd.DataFrame()


class PriceData:
    """Class for managing individual stock data"""
    
    def __init__(self, client: Optional[APIClient] = None):
        """
        Initialize stock data manager
        
        Args:
            client: EODHD API client (optional, will create a new one if None)
        """
        self.client = client if client else APIClient()
    
    def get_raw_price_data(self, symbol: str) -> pd.DataFrame:
        """
        Download raw price data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame containing price data
        """
        endpoint = f"eod/{symbol}"
        data = self.client.get(endpoint)
        return pd.DataFrame(data)
    
    def get_technical_data(self, symbol: str) -> pd.DataFrame:
        """
        Download split-adjusted technical data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame containing technical data
        """
        endpoint = f"technical/{symbol}"
        params = {"function": "splitadjusted"}
        data = self.client.get(endpoint, params)
        return pd.DataFrame(data)
        
    def get_historical_market_cap(self, symbol: str) -> pd.DataFrame:
        """
        Download historical market capitalization data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame containing market cap history
        """
        endpoint = f"historical-market-cap/{symbol}"
        data = self.client.get(endpoint)
        return pd.DataFrame(data)
    
    def process_price_data(self, 
                          raw_data_list: List[pd.DataFrame], 
                          technical_data_list: List[pd.DataFrame], 
                          tickers: List[str]) -> pd.DataFrame:
        """
        Process and combine raw and technical price data
        
        Args:
            raw_data_list: List of raw data DataFrames
            technical_data_list: List of technical data DataFrames
            tickers: List of corresponding symbols
            
        Returns:
            Combined price data DataFrame
        """
        final_price_list = []
        
        for idx, ticker in enumerate(tickers):
            tp = pd.DataFrame(technical_data_list[idx])
            rp = pd.DataFrame(raw_data_list[idx])
        
            if not rp.empty:
                fp = tp.merge(rp[['date', 'adjusted_close']], on='date', how='left')
                fp['ticker'] = ticker
                final_price_list.append(fp)
        
        if final_price_list:
            return pd.concat(final_price_list, ignore_index=True)
        return pd.DataFrame()
    
    def get_price_data_for_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Retrieve and combine price data for a list of symbols
        
        Args:
            tickers: List of symbols
            
        Returns:
            Combined price data DataFrame
        """
        raw_data = []
        technical_data = []
        
        for ticker in tickers:
            raw_data.append(self.get_raw_price_data(ticker))
            technical_data.append(self.get_technical_data(ticker))
        
        return self.process_price_data(raw_data, technical_data, tickers)


class FundamentalData:
    """Class for managing company fundamental data"""
    
    def __init__(self, client: Optional[APIClient] = None):
        """
        Initialize fundamental data manager
        
        Args:
            client: EODHD API client (optional, will create a new one if None)
        """
        self.client = client if client else APIClient()
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """
        Download fundamental data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing fundamental data
        """
        endpoint = f"fundamentals/{symbol}"
        return self.client.get(endpoint)
    
    def process_fundamental_data(self, 
                               fundamental_data_list: List[Dict], 
                               tickers: List[str], 
                               data_type: str) -> pd.DataFrame:
        """
        Process and extract specific fundamental data type
        
        Args:
            fundamental_data_list: List of fundamental data dictionaries
            tickers: List of corresponding symbols
            data_type: Type of data to extract ('general', 'Income_Statement', 
                      'Balance_Sheet', 'Cash_Flow', 'Earnings', 'outstandingShares')
            
        Returns:
            DataFrame containing extracted fundamental data
        """
        final_data_list = []
        
        for idx, ticker in enumerate(tickers):
            try:
                if data_type == "general":
                    filtered_data = {k: v for k, v in fundamental_data_list[idx]['General'].items() 
                                     if isinstance(v, str)}
                    tp = pd.DataFrame([filtered_data])
                    
                elif data_type == "Income_Statement":
                    tp = pd.DataFrame.from_dict(
                        fundamental_data_list[idx]['Financials']['Income_Statement']['quarterly'], 
                        orient='index'
                    )
                    
                elif data_type == "Balance_Sheet":
                    tp = pd.DataFrame.from_dict(
                        fundamental_data_list[idx]['Financials']['Balance_Sheet']['quarterly'], 
                        orient='index'
                    )
                    
                elif data_type == "Cash_Flow":
                    tp = pd.DataFrame.from_dict(
                        fundamental_data_list[idx]['Financials']['Cash_Flow']['quarterly'], 
                        orient='index'
                    )
                    
                elif data_type == "Earnings":
                    tp = pd.DataFrame.from_dict(
                        fundamental_data_list[idx]['Earnings']['History'], 
                        orient='index'
                    )
                    
                elif data_type == "outstandingShares":
                    tp = pd.DataFrame.from_dict(
                        fundamental_data_list[idx]['outstandingShares']['quarterly'], 
                        orient='index'
                    )
                    
                else:
                    raise ValueError(
                        "Invalid type. Use 'general', 'Income_Statement', 'Balance_Sheet', "
                        "'Cash_Flow', 'Earnings', or 'outstandingShares'."
                    )
                
                tp['ticker'] = ticker
                final_data_list.append(tp)
                
            except (KeyError, TypeError) as e:
                print(f"Error processing data for {ticker}: {e}")
        
        if final_data_list:
            return pd.concat(final_data_list, ignore_index=True)
        return pd.DataFrame()


class EODHDDataService:
    """
    Main service for accessing EODHD financial data.
    This class provides a unified interface for all functionalities.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize data service
        
        Args:
            api_key: EODHD API key
        """
        self.client = APIClient(api_key)
        self.exchange_data = ExchangeData(self.client)
        self.price_data = PriceData(self.client)
        self.fundamental_data = FundamentalData(self.client)
        self.index_data = IndexData(self.client)
    
    def get_sp500_historical_composition(self) -> pd.DataFrame:
        """
        Retrieve historical composition of the S&P 500
        
        Returns:
            DataFrame with historical composition
        """
        return self.index_data.get_historical_sp500()
    
    def get_sp500_components(self) -> pd.DataFrame:
        """
        Retrieve current components of the S&P 500
        
        Returns:
            DataFrame with current components
        """
        return self.index_data.get_sp500_components()
    
    def get_ticker_list_from_exchange(self, exchange_code: str) -> pd.DataFrame:
        """
        Retrieve list of stocks from an exchange
        
        Args:
            exchange_code: Exchange code
            
        Returns:
            DataFrame with list of stocks
        """
        return self.exchange_data.get_tickers_from_exchange(exchange_code)
    
    def get_price_data_for_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Retrieve and combine price data for a list of symbols
        
        Args:
            tickers: List of symbols
            
        Returns:
            Combined price data DataFrame
        """
        return self.price_data.get_price_data_for_tickers(tickers)
    
    def get_fundamental_data(self, tickers: List[str]) -> List:
        """
        Retrieve and process fundamental data for a list of symbols
        
        Args:
            tickers: List of symbols
            
        Returns:
            List of fundamental data dictionaries
        """
        fundamental_data = []
        
        for ticker in tickers:
            fundamental_data.append(self.fundamental_data.get_fundamental_data(ticker))
        
        return fundamental_data
    
    def process_fundamental_data(self, fundamental_data: List, data_type: str) -> pd.DataFrame:
        """
        Process and extract specific fundamental data type
        
        Args:
            fundamental_data: List of fundamental data dictionaries
            data_type: Type of fundamental data to extract
            
        Returns:
            DataFrame with extracted fundamental data
        """
        return self.fundamental_data.process_fundamental_data(fundamental_data, tickers, data_type)
    
    def get_sp500_price_data(self) -> pd.DataFrame:
        """
        Retrieve price data for the S&P 500 (via SPY)
        
        Returns:
            DataFrame with S&P 500 price data
        """
        return self.index_data.get_sp500_data()
