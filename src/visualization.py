import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Optional, Tuple
from statsmodels.tsa.stattools import acf, pacf


def plot_price_and_trend(
    df: pd.DataFrame,
    trend: Optional[pd.Series] = None,
    title: str = "Wykres cen i trendu"
) -> go.Figure:
    """
    Tworzy interaktywny wykres cen (świecowy lub liniowy) z nałożonym trendem.

    Parameters:
    -----------
    df : pd.DataFrame
        Ramka danych zawierająca ceny ('Open', 'High', 'Low', 'Close').
    trend : Optional[pd.Series]
        Szereg reprezentujący wyestymowany trend (np. liniowy).
    title : str
        Tytuł wykresu.

    Returns:
    --------
    go.Figure
        Wykres Plotly.
    """
    fig = go.Figure()

    # Wykres świecowy (Candlestick)
    if all(col in df.columns for col in ["Open", "High", "Low", "Close"]):
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Cena (OHLC)",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350"
            )
        )
    # Wykres liniowy (jeśli nie ma pełnych danych OHLC)
    elif "Close" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Cena Zamknięcia",
                line=dict(color="#2196f3", width=2)
            )
        )
    
    # Nałożenie trendu, jeśli jest podany
    if trend is not None and not trend.empty:
        fig.add_trace(
            go.Scatter(
                x=trend.index,
                y=trend,
                mode="lines",
                name="Trend OLS",
                line=dict(color="#ff9800", width=2.5, dash="dash")
            )
        )

    # Stylistyka wykresu
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="#ffffff")
        ),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            gridcolor="#2a2a2a",
            title="Data",
            tickfont=dict(color="#cccccc")
        ),
        yaxis=dict(
            gridcolor="#2a2a2a",
            title="Cena / Wartość",
            tickfont=dict(color="#cccccc")
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_acf_pacf(
    series: pd.Series,
    lags: int = 30
) -> Tuple[go.Figure, go.Figure]:
    """
    Generuje interaktywne wykresy funkcji autokorelacji (ACF) i autokorelacji cząstkowej (PACF).

    Parameters:
    -----------
    series : pd.Series
        Szereg czasowy do analizy.
    lags : int
        Liczba opóźnień do wizualizacji.

    Returns:
    --------
    Tuple[go.Figure, go.Figure]
        Dwie figury Plotly: (ACF, PACF).
    """
    clean_series = series.dropna()
    n_obs = len(clean_series)
    
    if n_obs < lags:
        lags = n_obs - 2

    acf_vals = acf(clean_series, nlags=lags, fft=True)
    pacf_vals = pacf(clean_series, nlags=lags, method="ols")
    
    ci_bound = 1.96 / np.sqrt(n_obs)
    lag_indices = np.arange(len(acf_vals))

    # 1. Wykres ACF
    fig_acf = go.Figure()
    fig_acf.add_scatter(
        x=[lag_indices[0], lag_indices[-1]],
        y=[ci_bound, ci_bound],
        mode="lines",
        line=dict(color="rgba(33, 150, 243, 0.2)", width=0),
        showlegend=False,
        name="Górna granica przedziału"
    )
    fig_acf.add_scatter(
        x=[lag_indices[0], lag_indices[-1]],
        y=[-ci_bound, -ci_bound],
        mode="lines",
        line=dict(color="rgba(33, 150, 243, 0.2)", width=0),
        fill="tonexty",
        fillcolor="rgba(33, 150, 243, 0.15)",
        name="Przedział istotności 95%",
        legendgroup="ci"
    )
    fig_acf.add_trace(
        go.Bar(
            x=lag_indices,
            y=acf_vals,
            marker_color="#2196f3",
            name="ACF",
            width=0.4
        )
    )
    fig_acf.add_trace(
        go.Scatter(
            x=lag_indices,
            y=acf_vals,
            mode="markers",
            marker=dict(color="#2196f3", size=6),
            showlegend=False
        )
    )
    fig_acf.update_layout(
        title=dict(text="Autokorelacja (ACF)", font=dict(size=16, color="#ffffff")),
        template="plotly_dark",
        xaxis=dict(gridcolor="#2a2a2a", title="Opóźnienie (Lag)", dtick=5),
        yaxis=dict(gridcolor="#2a2a2a", title="Współczynnik korelacji", range=[-1.1, 1.1]),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=True,
        legend=dict(x=0.8, y=0.9)
    )

    # 2. Wykres PACF
    fig_pacf = go.Figure()
    fig_pacf.add_scatter(
        x=[lag_indices[0], lag_indices[-1]],
        y=[ci_bound, ci_bound],
        mode="lines",
        line=dict(color="rgba(76, 175, 80, 0.2)", width=0),
        showlegend=False,
        name="Górna granica przedziału"
    )
    fig_pacf.add_scatter(
        x=[lag_indices[0], lag_indices[-1]],
        y=[-ci_bound, -ci_bound],
        mode="lines",
        line=dict(color="rgba(76, 175, 80, 0.2)", width=0),
        fill="tonexty",
        fillcolor="rgba(76, 175, 80, 0.15)",
        name="Przedział istotności 95%",
        legendgroup="ci"
    )
    fig_pacf.add_trace(
        go.Bar(
            x=lag_indices,
            y=pacf_vals,
            marker_color="#4caf50",
            name="PACF",
            width=0.4
        )
    )
    fig_pacf.add_trace(
        go.Scatter(
            x=lag_indices,
            y=pacf_vals,
            mode="markers",
            marker=dict(color="#4caf50", size=6),
            showlegend=False
        )
    )
    fig_pacf.update_layout(
        title=dict(text="Autokorelacja Cząstkowa (PACF)", font=dict(size=16, color="#ffffff")),
        template="plotly_dark",
        xaxis=dict(gridcolor="#2a2a2a", title="Opóźnienie (Lag)", dtick=5),
        yaxis=dict(gridcolor="#2a2a2a", title="Współczynnik korelacji cząstkowej", range=[-1.1, 1.1]),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=True,
        legend=dict(x=0.8, y=0.9)
    )

    return fig_acf, fig_pacf


def plot_forecast(
    historical: pd.Series,
    forecast: pd.DataFrame,
    title: str = "Prognoza modelu szeregów czasowych"
) -> go.Figure:
    """
    Rysuje historyczny szereg oraz prognozę punktową z cieniowanym przedziałem ufności 95%.
    Zawęża dane historyczne do ostatnich 100 dni w celu zachowania czytelności.

    Parameters:
    -----------
    historical : pd.Series
        Szereg czasowy danych historycznych (np. ceny).
    forecast : pd.DataFrame
        Ramka danych z prognozami (kolumny: 'mean', 'lower_ci', 'upper_ci').
    title : str
        Tytuł wykresu.

    Returns:
    --------
    go.Figure
        Interaktywny wykres Plotly.
    """
    fig = go.Figure()

    # Zawężenie historii do ostatnich 100 sesji
    hist_sliced = historical.dropna().tail(100)

    # 1. Linia danych historycznych
    fig.add_trace(
        go.Scatter(
            x=hist_sliced.index,
            y=hist_sliced.values,
            mode="lines",
            name="Historia (ostatnie 100 sesji)",
            line=dict(color="#2196f3", width=2)
        )
    )

    # Połączenie ostatniej wartości historycznej z pierwszą wartością prognozy (wizualny pomost)
    # Tworzymy serie łączone do rysowania prognozy
    fc_index = forecast.index
    fc_mean = forecast["mean"].values
    fc_lower = forecast["lower_ci"].values
    fc_upper = forecast["upper_ci"].values

    bridge_x = [hist_sliced.index[-1]] + list(fc_index)
    bridge_mean = [hist_sliced.values[-1]] + list(fc_mean)
    bridge_lower = [hist_sliced.values[-1]] + list(fc_lower)
    bridge_upper = [hist_sliced.values[-1]] + list(fc_upper)

    # 2. Przedział ufności (górna granica)
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Górna granica ufności",
            legendgroup="ci_forecast"
        )
    )

    # 3. Przedział ufności (dolna granica z cieniowaniem)
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 152, 0, 0.15)",
            name="Przedział ufności 95%",
            legendgroup="ci_forecast"
        )
    )

    # 4. Linia prognozy punktowej (średnia)
    fig.add_trace(
        go.Scatter(
            x=bridge_x,
            y=bridge_mean,
            mode="lines+markers",
            name="Prognoza (Średnia)",
            line=dict(color="#ff9800", width=2.5),
            marker=dict(size=4)
        )
    )

    # Stylistyka wykresu
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="#ffffff")
        ),
        template="plotly_dark",
        xaxis=dict(
            gridcolor="#2a2a2a",
            title="Data",
            tickfont=dict(color="#cccccc")
        ),
        yaxis=dict(
            gridcolor="#2a2a2a",
            title="Wartość szeregu",
            tickfont=dict(color="#cccccc")
        ),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig
