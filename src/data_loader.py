import pandas as pd
import numpy as np
import yfinance as yf
import requests
from typing import Optional, List, Dict, Any


def fetch_stock_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Pobiera historyczne dane giełdowe dla podanego tickera z API Yahoo Finance.
    Usuwa strefę czasową (tz-naive) w celu zapewnienia kompatybilności z modelami statsmodels.

    Parameters:
    -----------
    ticker : str
        Symbol giełdowy aktywa (np. 'AAPL', 'GPW:CDR').
    start_date : str
        Data początkowa w formacie 'YYYY-MM-DD'.
    end_date : str
        Data końcowa w formacie 'YYYY-MM-DD'.

    Returns:
    --------
    pd.DataFrame
        Ramka danych zawierająca pobrane notowania (OHLCV).

    Raises:
    -------
    ValueError
        Jeśli pobrana ramka danych jest pusta lub wystąpi błąd podczas pobierania.
    """
    if not ticker:
        raise ValueError("Symbol giełdowy (ticker) nie może być pusty.")
    
    try:
        # Pobieranie danych z yfinance
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    except Exception as e:
        raise ValueError(f"Wystąpił błąd sieciowy podczas komunikacji z API Yahoo Finance: {str(e)}")

    if df.empty:
        raise ValueError(
            f"Brak danych dla symbolu '{ticker}' w przedziale od {start_date} do {end_date}. "
            f"Upewnij się, że symbol jest poprawny oraz że rynek był otwarty w tym okresie."
        )
    
    # Jeśli yfinance zwraca MultiIndex, spłaszczamy go
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Upewniamy się, że indeks to DatetimeIndex i usuwamy strefę czasową
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
        
    return df


def calculate_returns(
    df: pd.DataFrame,
    target_col: str = "Close",
    method: str = "log"
) -> pd.DataFrame:
    """
    Oblicza proste lub logarytmiczne stopy zwrotu z cen zamknięcia i dodaje kolumnę 'Returns'.

    Parameters:
    -----------
    df : pd.DataFrame
        Ramka danych z notowaniami historycznymi zawierająca kolumnę `target_col`.
    target_col : str
        Nazwa kolumny z cenami (domyślnie 'Close').
    method : str
        Metoda obliczania: 'log' (logarytmiczne) lub 'simple' (proste).

    Returns:
    --------
    pd.DataFrame
        Kopia ramki danych rozszerzona o kolumnę 'Returns'.
    """
    if target_col not in df.columns:
        raise KeyError(f"Kolumna '{target_col}' nie istnieje w przekazanej ramce danych.")
    
    df_copy = df.copy()
    prices = df_copy[target_col].astype(float)

    if method == "log":
        df_copy["Returns"] = np.log(prices).diff()
    elif method == "simple":
        df_copy["Returns"] = prices.pct_change()
    else:
        raise ValueError("Nieznana metoda obliczania stóp zwrotu. Wybierz 'log' lub 'simple'.")
        
    return df_copy


def search_symbols(query: str) -> List[Dict[str, Any]]:
    """
    Wyszukuje symbole giełdowe, nazwy firm oraz indeksy za pomocą API wyszukiwania Yahoo Finance.

    Parameters:
    -----------
    query : str
        Fraza kluczowa wpisana przez użytkownika (np. 'Apple', 'S&P 500', 'CDR').

    Returns:
    --------
    List[Dict[str, Any]]
        Lista słowników reprezentujących dopasowane instrumenty (klucze: symbol, shortname, exchange itp.).
    """
    if not query or len(query.strip()) < 2:
        return []

    # Kodujemy frazę do formatu URL
    encoded_query = requests.utils.quote(query.strip())
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={encoded_query}"
    
    # Przeglądarkowy User-Agent jest wymagany, inaczej Yahoo blokuje zapytanie (403 Forbidden)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("quotes", [])
    except Exception:
        # W razie błędu sieciowego zwracamy pustą listę
        pass
        
    return []
