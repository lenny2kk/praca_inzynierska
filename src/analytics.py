import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from statsmodels.stats.diagnostic import acorr_ljungbox


def estimate_trend(
    df: pd.DataFrame,
    target_col: str = "Close"
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Estymuje trend liniowy dla szeregu czasowego przy użyciu OLS (metoda najmniejszych kwadratów).

    Zmienną objaśnianą (Y) są ceny lub stopy zwrotu.
    Zmienną objaśniającą (X) jest sekwencja czasu t = 1, 2, ..., T.

    Parameters:
    -----------
    df : pd.DataFrame
        Ramka danych zawierająca szereg czasowy z indeksem typu Datetime.
    target_col : str
        Nazwa kolumny, dla której estymowany jest trend.

    Returns:
    --------
    Tuple[pd.Series, Dict[str, float]]
        - Szereg czasowy trendu.
        - Słownik z metrykami: 'r2' (R^2), 'slope' (współczynnik kierunkowy/przyrost),
          'intercept' (wyraz wolny).
    """
    if target_col not in df.columns:
        raise KeyError(f"Kolumna '{target_col}' nie istnieje w ramce danych.")
        
    series = df[target_col].dropna()
    
    if len(series) < 2:
        raise ValueError("Za mało obserwacji (minimum 2), aby wyestymować trend.")

    t = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(t, y)

    trend_values = model.predict(t).flatten()
    trend_series = pd.Series(trend_values, index=series.index)

    metrics = {
        "r2": float(model.score(t, y)),
        "slope": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0])
    }

    return trend_series, metrics


def check_stationarity(
    series: pd.Series
) -> Dict[str, Any]:
    """
    Przeprowadza Rozszerzony Test Dickeya-Fullera (ADF) badający stacjonarność szeregu czasowego.

    Parameters:
    -----------
    series : pd.Series
        Szereg czasowy do zbadania.

    Returns:
    --------
    Dict[str, Any]
        Słownik z wynikami testu ADF.
    """
    clean_series = series.dropna()

    if len(clean_series) < 10:
        raise ValueError("Szereg czasowy jest zbyt krótki (minimum 10 obserwacji) do wykonania testu ADF.")

    result = adfuller(clean_series)

    p_value = float(result[1])
    is_stationary = p_value < 0.05

    results_dict = {
        "adf_statistic": float(result[0]),
        "p_value": p_value,
        "lags_used": int(result[2]),
        "n_obs": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "is_stationary": is_stationary
    }

    return results_dict


def fit_arima_model(
    series: pd.Series,
    order: Tuple[int, int, int],
    seasonal_order: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[SARIMAXResults, Dict[str, Any]]:
    """
    Dopasowuje model ARIMA/SARIMA do szeregu czasowego przy użyciu SARIMAX z statsmodels.

    Parametry stacjonarności i odwracalności (enforce_stationarity, enforce_invertibility)
    są wyłączone w celu zapewnienia stabilności dopasowania w interaktywnym UI dla dowolnych p, d, q.

    Parameters:
    -----------
    series : pd.Series
        Szereg czasowy (ceny lub stopy zwrotu).
    order : Tuple[int, int, int]
        Krotka (p, d, q) określająca:
        - p: rząd autoregresji (AR)
        - d: rząd różnicowania (I)
        - q: rząd średniej ruchomej (MA)
    seasonal_order : Optional[Tuple[int, int, int, int]]
        Krotka (P, D, Q, s) dla modelu sezonowego (SARIMA).

    Returns:
    --------
    Tuple[SARIMAXResults, Dict[str, Any]]
        - Obiekt wyników dopasowania statsmodels (SARIMAXResults).
        - Słownik z kryteriami informacyjnymi (AIC, BIC, LLF) oraz wynikami testów diagnostycznych
          residuów (Ljung-Box na autokorelację, Jarque-Bera na normalność rozkładu).
    """
    # Oczyszczenie szeregu z wartości NaN
    clean_series = series.dropna()
    
    if len(clean_series) < 15:
        raise ValueError("Szereg czasowy jest zbyt krótki (wymagane min. 15 obserwacji) do estymacji modelu ARIMA.")

    # Definicja i dopasowanie modelu SARIMAX
    model = SARIMAX(
        clean_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    # Optymalizacja (ukrywamy logi w konsoli)
    results = model.fit(disp=False)
    
    # Wyznaczenie kryteriów informacyjnych
    aic = float(results.aic)
    bic = float(results.bic)
    llf = float(results.llf)

    # 1. Diagnostyka reszt: Test Ljunga-Boxa (H0: brak autokorelacji w resztach / biały szum)
    # Wybieramy opóźnienie równe 10 (lub mniejsze, jeśli szereg jest bardzo krótki)
    lb_lag = min(10, len(clean_series) // 5)
    lb_lag = max(1, lb_lag) # upewniamy się, że wynosi co najmniej 1
    
    lb_df = acorr_ljungbox(results.resid, lags=[lb_lag], return_df=True)
    lb_stat = float(lb_df[f"lb_stat"].iloc[0])
    lb_pvalue = float(lb_df[f"lb_pvalue"].iloc[0])

    # 2. Diagnostyka reszt: Test Jarque-Bera (H0: reszty mają rozkład normalny)
    jb_results = results.test_normality(method="jarquebera")
    jb_stat = float(jb_results[0, 0])
    jb_pvalue = float(jb_results[0, 1])

    metrics = {
        "aic": aic,
        "bic": bic,
        "llf": llf,
        "ljung_box_lag": lb_lag,
        "ljung_box_stat": lb_stat,
        "ljung_box_pvalue": lb_pvalue,
        "jarque_bera_stat": jb_stat,
        "jarque_bera_pvalue": jb_pvalue
    }

    return results, metrics


def generate_forecast(
    model_result: SARIMAXResults,
    steps: int
) -> pd.DataFrame:
    """
    Generuje prognozę punktową oraz przedziały ufności 95% dla dopasowanego modelu SARIMAX.
    Tworzy przyszłe indeksy dat biznesowych (dni robocze) na podstawie ostatniej daty w danych.

    Parameters:
    -----------
    model_result : SARIMAXResults
        Obiekt dopasowanego modelu SARIMAX z statsmodels.
    steps : int
        Liczba sesji giełdowych (dni roboczych) w przód do prognozowania.

    Returns:
    --------
    pd.DataFrame
        Ramka danych z prognozami, zawierająca kolumny:
        - 'mean': prognoza punktowa (średnia warunkowa)
        - 'lower_ci': dolna granica 95% przedziału ufności
        - 'upper_ci': górna granica 95% przedziału ufności
        Indeksem ramki są wygenerowane przyszłe daty biznesowe.
    """
    if steps <= 0:
        raise ValueError("Liczba kroków prognozy musi być większa od zera.")

    # Generowanie prognozy z statsmodels
    forecast_obj = model_result.get_forecast(steps=steps)
    mean_forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int(alpha=0.05) # 95% przedział ufności

    # Pobranie indeksu oryginalnego szeregu czasowego
    orig_index = model_result.model.data.row_labels
    
    # Wyznaczenie przyszłych dat (tylko dni robocze)
    last_date = pd.to_datetime(orig_index[-1])
    future_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    # Konstrukcja wynikowej ramki danych
    forecast_df = pd.DataFrame(index=future_index)
    forecast_df["mean"] = mean_forecast.values
    forecast_df["lower_ci"] = conf_int.iloc[:, 0].values
    forecast_df["upper_ci"] = conf_int.iloc[:, 1].values

    return forecast_df
