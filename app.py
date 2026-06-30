import streamlit as st
import pandas as pd
from src.data_loader import fetch_stock_data, calculate_returns, search_symbols
from src.analytics import estimate_trend, check_stationarity, fit_arima_model, generate_forecast
from src.visualization import plot_price_and_trend, plot_acf_pacf, plot_forecast

# Ustawienia strony Streamlit
st.set_page_config(
    page_title="Analiza Szeregów Czasowych i Trendów Giełdowych",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Stylizacja CSS dla estetycznego wyglądu (modern dark dashboard)
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Analiza ilościowa i prognozowanie szeregów czasowych")
st.markdown(
    """
    Aplikacja wspiera proces analizy stacjonarności, detekcji trendów metodą najmniejszych kwadratów (OLS)
    oraz modelowania i prognozowania za pomocą modeli klasy **ARIMA/SARIMA**.
    """
)

# Inicjalizacja stanu sesji dla przechowywania danych i wyników modeli
if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
if "ticker_name" not in st.session_state:
    st.session_state.ticker_name = ""
if "model_fit_results" not in st.session_state:
    st.session_state.model_fit_results = None

# Panel boczny - Parametry wejściowe
st.sidebar.header("⚙️ Ustawienia Wejściowe")

# Wyszukiwanie symbolu z podpowiedziami (Autocomplete)
search_query = st.sidebar.text_input("Wyszukaj firmę / symbol (np. Apple, S&P, GPW)", value="AAPL")

ticker = ""
if len(search_query.strip()) >= 2:
    with st.sidebar:
        try:
            # Pobieranie propozycji z Yahoo Finance API
            suggestions = search_symbols(search_query)
            if suggestions:
                options = []
                for q in suggestions:
                    symbol = q.get("symbol")
                    name = q.get("shortname") or q.get("longname") or "Brak nazwy"
                    exchange = q.get("exchDisp") or q.get("exchange") or "Inna"
                    type_disp = q.get("typeDisp") or "Instrument"
                    if symbol:
                        label = f"{symbol} - {name} ({exchange} | {type_disp})"
                        options.append((symbol, label))
                
                # Zwrócenie listy wyboru
                selected_ticker = st.selectbox(
                    "Wybierz instrument z wyników:",
                    options=options,
                    format_func=lambda x: x[1]
                )
                if selected_ticker:
                    ticker = selected_ticker[0]
            else:
                st.caption("Brak podpowiedzi. Użyję dokładnie wpisanego tekstu.")
                ticker = search_query.strip().upper()
        except Exception:
            ticker = search_query.strip().upper()
else:
    ticker = search_query.strip().upper()

# Czyszczenie stanu sesji, jeśli zmieniono ticker
if ticker != st.session_state.ticker_name:
    st.session_state.stock_data = None
    st.session_state.model_fit_results = None

# Wybór zakresu dat z walidacją
col_date1, col_date2 = st.sidebar.columns(2)
with col_date1:
    start_date = st.date_input("Data początkowa", pd.to_datetime("2023-01-01"))
with col_date2:
    end_date = st.date_input("Data końcowa", pd.to_datetime("today"))

returns_method = st.sidebar.selectbox(
    "Metoda stóp zwrotu",
    options=["log", "simple"],
    format_func=lambda x: "Logarytmiczne" if x == "log" else "Zwykłe (proste)"
)

st.sidebar.markdown("---")
st.sidebar.header("📉 Parametry Diagnostyczne")
show_trend = st.sidebar.checkbox("Pokaż trend liniowy OLS", value=False)
lags = st.sidebar.slider("Liczba opóźnień (ACF/PACF)", min_value=10, max_value=50, value=30)

# Przycisk pobierania danych
fetch_button = st.sidebar.button("Pobierz i analizuj dane", use_container_width=True)

# Wywołanie pobierania po kliknięciu przycisku
if fetch_button:
    if not ticker:
        st.sidebar.error("Błąd: Symbol giełdowy nie może być pusty.")
    elif start_date >= end_date:
        st.sidebar.error("Błąd: Data początkowa musi być wcześniejsza niż końcowa.")
    else:
        with st.spinner(f"Pobieranie danych dla {ticker}..."):
            try:
                # 1. Pobieranie danych
                df = fetch_stock_data(
                    ticker=ticker,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                # 2. Obliczanie stóp zwrotu
                df = calculate_returns(df, target_col="Close", method=returns_method)
                
                # Zapis do stanu sesji
                st.session_state.stock_data = df
                st.session_state.ticker_name = ticker
                st.session_state.model_fit_results = None  # Resetujemy stary model po pobraniu nowych danych
                st.toast(f"Pomyślnie załadowano dane dla {st.session_state.ticker_name}!", icon="✅")
            except Exception as e:
                st.error(f"Nie udało się pobrać danych dla symbolu '{ticker}': {str(e)}")
                st.session_state.stock_data = None
                st.session_state.model_fit_results = None

# Główny interfejs aplikacji
if st.session_state.stock_data is not None:
    df = st.session_state.stock_data
    ticker_name = st.session_state.ticker_name

    # Obliczenie metryk do nagłówka
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else last_row
    
    last_price = float(last_row["Close"])
    prev_price = float(prev_row["Close"])
    price_diff = last_price - prev_price
    price_pct = (price_diff / prev_price) * 100

    # Sekcja wskaźników (Metrics)
    st.markdown("### 📊 Kluczowe Wskaźniki")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric(
        label=f"Cena Zamknięcia ({ticker_name})",
        value=f"{last_price:.2f} USD",
        delta=f"{price_diff:+.2f} ({price_pct:+.2f}%)"
    )
    
    # Ostatnia stopa zwrotu
    last_ret = last_row["Returns"]
    m_col2.metric(
        label="Ostatnia Stopa Zwrotu",
        value="Brak danych" if pd.isna(last_ret) else f"{last_ret * 100:.3f}%"
    )
    
    # Wolumen
    last_vol = int(last_row["Volume"])
    m_col3.metric(
        label="Wolumen sesji",
        value=f"{last_vol:,}"
    )
    
    # Liczba obserwacji
    m_col4.metric(
        label="Liczba obserwacji",
        value=str(len(df))
    )

    st.markdown("---")

    # Estymacja trendu liniowego OLS
    trend_series = None
    trend_metrics = None
    if show_trend:
        try:
            trend_series, trend_metrics = estimate_trend(df, target_col="Close")
        except Exception as e:
            st.error(f"Błąd estymacji trendu: {str(e)}")

    # PODZIAŁ 1: Ceny i Statystyki
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📈 Interaktywny Wykres Cenowy")
        fig = plot_price_and_trend(
            df, 
            trend=trend_series, 
            title=f"Wykres notowań {ticker_name}" + (" z trendem OLS" if show_trend else "")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📋 Podgląd Danych i Analiza Statystyczna")
        tabs = st.tabs([
            "Tabela danych", 
            "Statystyki opisowe", 
            "Test stacjonarności (ADF)", 
            "Analiza trendu OLS"
        ])
        
        # TAB 1: Tabela Danych
        with tabs[0]:
            st.dataframe(
                df[["Open", "High", "Low", "Close", "Volume", "Returns"]].tail(10),
                use_container_width=True
            )
            st.caption("Ostatnie 10 rekordów z szeregu czasowego.")
            
        # TAB 2: Statystyki Opisowe
        with tabs[1]:
            st.dataframe(
                df[["Close", "Returns"]].describe(),
                use_container_width=True
            )
            st.caption("Podstawowe parametry statystyczne rozkładu cen i stóp zwrotu.")
            
        # TAB 3: Test ADF
        with tabs[2]:
            st.markdown("#### Rozszerzony Test Dickeya-Fullera (ADF)")
            adf_target = st.selectbox(
                "Wybierz szereg do analizy stacjonarności:",
                options=["Close", "Returns"],
                format_func=lambda x: "Cena Zamknięcia (Close)" if x == "Close" else "Stopa Zwrotu (Returns)",
                key="adf_select"
            )
            
            try:
                adf_res = check_stationarity(df[adf_target])
                
                adf_data = {
                    "Statystyka testowa ADF": f"{adf_res['adf_statistic']:.4f}",
                    "p-value": f"{adf_res['p_value']:.4e}",
                    "Liczba użytych opóźnień": str(adf_res['lags_used']),
                    "Liczba obserwacji": str(adf_res['n_obs']),
                    "Wartość krytyczna (1%)": f"{adf_res['critical_values']['1%']:.4f}",
                    "Wartość krytyczna (5%)": f"{adf_res['critical_values']['5%']:.4f}",
                    "Wartość krytyczna (10%)": f"{adf_res['critical_values']['10%']:.4f}"
                }
                
                st.table(pd.Series(adf_data, name="Wartość"))
                
                if adf_res['is_stationary']:
                    st.success(
                        f"**Decyzja:** Szereg jest **stacjonarny** (odrzucamy hipotezę zerową H0 przy poziomie istotności 5%, "
                        f"ponieważ p-value = {adf_res['p_value']:.4f} < 0.05).\n\n"
                        f"Szereg spełnia podstawowe założenie modeli klasy ARIMA."
                    )
                else:
                    st.warning(
                        f"**Decyzja:** Szereg jest **niestacjonarny** (brak podstaw do odrzucenia hipotezę zerowej H0, "
                        f"ponieważ p-value = {adf_res['p_value']:.4f} >= 0.05).\n\n"
                        f"**Wskazówka ekonometryczna:** Przed modelowaniem ARIMA szereg ten musi zostać poddany "
                        f"różnicowaniu (differencing), np. poprzez przejście do stóp zwrotu lub obliczenie pierwszych różnic."
                    )
            except Exception as e:
                st.error(f"Nie można wykonać testu ADF: {str(e)}")

        # TAB 4: Regresja OLS
        with tabs[3]:
            st.markdown("#### Detekcja trendu liniowego metodą najmniejszych kwadratów (OLS)")
            st.markdown(
                "Dopasowujemy równanie trendu postaci:  \n"
                "$$\hat{Y}_t = \\beta_0 + \\beta_1 \\cdot t$$"
            )
            
            try:
                temp_trend_series, temp_trend_metrics = estimate_trend(df, target_col="Close")
                
                st.markdown(
                    f"**Wyestymowane równanie trendu cenowego:**  \n"
                    f"$$\\hat{{Price}}_t = {temp_trend_metrics['intercept']:.2f} + {temp_trend_metrics['slope']:.4f} \\cdot t$$"
                )
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric(
                    label="Współczynnik kierunkowy (Slope / Beta 1)",
                    value=f"{temp_trend_metrics['slope']:.4f}"
                )
                col_m2.metric(
                    label="Współczynnik determinacji (R^2)",
                    value=f"{temp_trend_metrics['r2'] * 100:.2f}%"
                )
                
                st.info(
                    f"**Interpretacja:** Średni przyrost ceny w badanym okresie wynosi **{temp_trend_metrics['slope']:.4f} USD** "
                    f"na jedną sesję giełdową. Liniowy upływ czasu wyjaśnia **{temp_trend_metrics['r2'] * 100:.2f}%** wariancji ceny zamknięcia."
                )
            except Exception as e:
                st.error(f"Błąd estymacji trendu: {str(e)}")

    st.markdown("---")

    # PODZIAŁ 2: Korelogramy (ACF / PACF)
    st.subheader("🔍 Analiza Autokorelacji (Metodologia Boxa-Jenkinsa)")
    st.markdown(
        """
        Wykresy funkcji **autokorelacji (ACF)** oraz **autokorelacji cząstkowej (PACF)** służą do wstępnej identyfikacji 
        rządów opóźnień dla modeli klasy ARIMA(p, d, q).
        """
    )
    
    acf_pacf_target = st.selectbox(
        "Wybierz szereg do analizy korelacji:",
        options=["Close", "Returns"],
        format_func=lambda x: "Cena Zamknięcia (Close - szereg surowy)" if x == "Close" else "Stopa Zwrotu (Returns - szereg zróżnicowany)",
        key="acf_pacf_select"
    )

    try:
        col_acf, col_pacf = st.columns(2)
        fig_acf, fig_pacf = plot_acf_pacf(df[acf_pacf_target], lags=lags)
        
        with col_acf:
            st.plotly_chart(fig_acf, use_container_width=True)
        with col_pacf:
            st.plotly_chart(fig_pacf, use_container_width=True)
            
        if acf_pacf_target == "Close":
            st.info(
                "💡 **Obserwacja (Ceny):** ACF wygasza bardzo powoli (liniowo), a pierwsze opóźnienie PACF jest bliskie 1. "
                "To klasyczny objaw niestacjonarności. Wymagane jest różnicowanie."
            )
        else:
            st.success(
                "💡 **Obserwacja (Stopy Zwrotu):** ACF i PACF gwałtownie wygasają. "
                "Możesz teraz zidentyfikować potencjalne rzędy modeli (np. AR(p) lub MA(q))."
            )
    except Exception as e:
        st.error(f"Błąd podczas generowania wykresów ACF/PACF: {str(e)}")

    st.markdown("---")

    # PODZIAŁ 3: Modelowanie i Prognozowanie ARIMA/SARIMA
    st.subheader("🔮 Modelowanie i Prognozowanie ARIMA/SARIMA")
    st.markdown(
        """
        W tej sekcji możesz dopasować model **ARIMA(p, d, q)** lub jego sezonowy wariant **SARIMA(p, d, q)x(P, D, Q)s** 
        i prognozować przyszłe notowania.
        """
    )

    # Parametry modelu w formularzu dwukolumnowym
    model_col1, model_col2 = st.columns([1, 1])

    with model_col1:
        st.markdown("##### Parametry Główne (ARIMA)")
        
        arima_target = st.selectbox(
            "Szereg do modelowania:",
            options=["Close", "Returns"],
            format_func=lambda x: "Cena Zamknięcia (Close)" if x == "Close" else "Stopa Zwrotu (Returns)",
            key="arima_target_select"
        )
        
        # Domyślny dobór d w zależności od szeregu
        default_d = 1 if arima_target == "Close" else 0
        
        col_p, col_d, col_q = st.columns(3)
        p_param = col_p.number_input("Rząd AR (p)", min_value=0, max_value=10, value=1, step=1)
        d_param = col_d.number_input("Rząd różnicowania (d)", min_value=0, max_value=2, value=default_d, step=1)
        q_param = col_q.number_input("Rząd MA (q)", min_value=0, max_value=10, value=1, step=1)

        forecast_steps = st.slider("Horyzont prognozy (dni robocze)", min_value=5, max_value=60, value=21)

    with model_col2:
        st.markdown("##### Parametry Sezonowe (SARIMA)")
        enable_seasonal = st.checkbox("Włącz komponent sezonowy", value=False)
        
        if enable_seasonal:
            col_P, col_D, col_Q, col_s = st.columns(4)
            P_param = col_P.number_input("Sezonowe AR (P)", min_value=0, max_value=5, value=0, step=1)
            D_param = col_D.number_input("Sezonowe I (D)", min_value=0, max_value=2, value=0, step=1)
            Q_param = col_Q.number_input("Sezonowe MA (Q)", min_value=0, max_value=5, value=0, step=1)
            s_param = col_s.number_input("Długość sezonu (s)", min_value=2, max_value=252, value=5, step=1,
                                         help="Liczba obserwacji na cykl sezonowy (np. 5 dla sezonowości tygodniowej).")
        else:
            P_param, D_param, Q_param, s_param = 0, 0, 0, 0

    fit_model_button = st.button("🚀 Estymuj Model i Prognozuj", use_container_width=True)

    # Uruchomienie modelowania i zapis wyników do stanu sesji
    if fit_model_button:
        seasonal_order = (P_param, D_param, Q_param, s_param) if enable_seasonal else None
        
        with st.spinner("Estymacja modelu SARIMAX w toku..."):
            try:
                # Estymacja modelu
                model_res, metrics = fit_arima_model(
                    series=df[arima_target],
                    order=(p_param, d_param, q_param),
                    seasonal_order=seasonal_order
                )
                
                # Generowanie prognozy
                forecast_df = generate_forecast(model_res, steps=forecast_steps)
                
                # Zapisujemy do stanu sesji
                st.session_state.model_fit_results = {
                    "metrics": metrics,
                    "forecast_df": forecast_df,
                    "arima_target": arima_target,
                    "order": (p_param, d_param, q_param),
                    "seasonal_order": seasonal_order
                }
                st.toast("Model wyestymowany pomyślnie!", icon="⚡")
            except Exception as e:
                st.error(f"Błąd podczas estymacji modelu: {str(e)}")
                st.session_state.model_fit_results = None

    # Wyświetlanie wyników modelu, jeżeli są w pamięci sesji
    if st.session_state.model_fit_results is not None:
        model_data = st.session_state.model_fit_results
        metrics = model_data["metrics"]
        forecast_df = model_data["forecast_df"]
        target_name = model_data["arima_target"]
        order_str = f"ARIMA{model_data['order']}"
        if model_data["seasonal_order"]:
            order_str += f" x {model_data['seasonal_order']}"

        st.markdown(f"#### 📊 Wyniki dla modelu: `{order_str}` na szeregu `{target_name}`")
        
        # 1. Metryki Jakości Dopasowania
        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Akaike IC (AIC)", f"{metrics['aic']:.2f}")
        res_col2.metric("Bayesian IC (BIC)", f"{metrics['bic']:.2f}")
        res_col3.metric("Log Likelihood (LLF)", f"{metrics['llf']:.2f}")

        # 2. Wykres prognozy oraz diagnostyka reszt
        layout_col1, layout_col2 = st.columns([3, 2])

        with layout_col1:
            st.markdown("##### 📈 Wykres Prognozy i Przedziału Ufności")
            
            historical_series = df[target_name]
            fig_fc = plot_forecast(
                historical=historical_series,
                forecast=forecast_df,
                title=f"Prognoza modelu {order_str} dla {ticker_name} ({target_name})"
            )
            st.plotly_chart(fig_fc, use_container_width=True)

        with layout_col2:
            st.markdown("##### 🔬 Diagnostyka Reszt Modelu")
            
            # Weryfikacja autokorelacji
            lb_p = metrics["ljung_box_pvalue"]
            if lb_p > 0.05:
                st.success(
                    f"✅ **Test Ljunga-Boxa (lag={metrics['ljung_box_lag']}):**  \n"
                    f"Statystyka: **{metrics['ljung_box_stat']:.4f}**, p-value = **{lb_p:.4f}** > 0.05.  \n"
                    f"Brak podstaw do odrzucenia H0. Reszty modelu **są białym szumem** (brak istotnej autokorelacji), "
                    f"co oznacza, że model poprawnie wychwycił zależności czasowe."
                )
            else:
                st.warning(
                    f"⚠️ **Test Ljunga-Boxa (lag={metrics['ljung_box_lag']}):**  \n"
                    f"Statystyka: **{metrics['ljung_box_stat']:.4f}**, p-value = **{lb_p:.4f}** <= 0.05.  \n"
                    f"Odrzucamy H0. W resztach **nadal występuje autokorelacja**. Model nie wychwycił wszystkich informacji. "
                    f"Spróbuj zwiększyć rząd p lub q."
                )

            # Weryfikacja normalności reszt
            jb_p = metrics["jarque_bera_pvalue"]
            if jb_p > 0.05:
                st.success(
                    f"✅ **Test Jarque-Bera:**  \n"
                    f"Statystyka: **{metrics['jarque_bera_stat']:.2f}**, p-value = **{jb_p:.4f}** > 0.05.  \n"
                    f"Brak podstaw do odrzucenia H0. Reszty mają **rozkład zbliżony do normalnego**."
                )
            else:
                st.info(
                    f"ℹ️ **Test Jarque-Bera:**  \n"
                    f"Statystyka: **{metrics['jarque_bera_stat']:.2f}**, p-value = **{jb_p:.4e}** <= 0.05.  \n"
                    f"Odrzucamy H0. Reszty **nie mają rozkładu normalnego** (typowe zjawisko dla szeregów finansowych "
                    f"ze względu na efekt grubych ogonów i występowanie wartości skrajnych)."
                )

        # 3. Tabela prognozy
        st.markdown("##### 📋 Tabela prognozowanych wartości")
        st.dataframe(
            forecast_df.style.format("{:.4f}"),
            use_container_width=True
        )
        st.caption(
            "Przedziały ufności dla każdego kroku w przód określają granice, w których rzeczywista cena/stopa zwrotu "
            "powinna znaleźć się z prawdopodobieństwem 95%. Zauważ, że przedziały rozszerzają się w czasie, co odzwierciedla "
            "rosnący poziom niepewności."
        )

else:
    st.info("💡 Użyj panelu bocznego, aby wyszukać symbol giełdowy i kliknąć przycisk **Pobierz i analizuj dane**.")
