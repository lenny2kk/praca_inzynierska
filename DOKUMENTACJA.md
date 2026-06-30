# Dokumentacja Techniczno-Metodologiczna Projektu

**Temat pracy:** „Aplikacja do ilościowej analizy danych giełdowych i estymacji trendów na podstawie modeli szeregów czasowych”  
**Stos technologiczny:** Python 3, Streamlit, yfinance, Pandas, NumPy, Plotly, Statsmodels, Scikit-learn, Requests.  
**Autor dokumentacji:** Antigravity (AI Coding Assistant & Econometric Analyst)

---

## 1. Architektura Systemu i Podział Modularny

Aplikacja została zaprojektowana zgodnie z pryncypiami czystego kodu (**clean code**) oraz zasadą separacji obaw (**Separation of Concerns**). Podział na warstwy logiczne umożliwia łatwe rozbudowywanie kodu, a studentowi ułatwia dokumentację w rozdziale technicznym pracy dyplomowej.

```
praca inzynierska/
├── requirements.txt            # Zbiór zależności (zarządzanie środowiskiem)
├── app.py                      # Główny plik Streamlit sterujący interfejsem użytkownika
├── DOKUMENTACJA.md             # Niniejszy opis projektu
└── src/                        # Pakiet źródłowy aplikacji (Python Package)
    ├── __init__.py             # Plik inicjalizacyjny pakietu
    ├── data_loader.py          # Pobieranie, walidacja i transformacje danych
    ├── analytics.py            # Testy statystyczne i silnik modelowania ARIMA/SARIMA
    └── visualization.py        # Generowanie interaktywnych wykresów w standardzie Plotly
```

---

## 2. Opis Funkcjonalny Modułów (Co? Jak? Dlaczego?)

### A. Pozyskiwanie danych i Autocomplete (`src/data_loader.py`)

Moduł odpowiada za pobieranie szeregów czasowych z rynku giełdowego oraz za dynamiczne wyszukiwanie instrumentów finansowych.

#### Funkcje składowe:
* **`fetch_stock_data(ticker, start_date, end_date)`**: Pobiera historyczne ceny aktywów (OHLCV).
  * *Jak działa:* Wywołuje bibliotekę `yfinance`. Następnie spłaszcza ewentualny MultiIndex kolumnowy, konwertuje indeks na typ `DatetimeIndex` i usuwa informacje o strefach czasowych (`tz_localize(None)`).
  * *Dlaczego tak:* Modele klasy ARIMA z biblioteki `statsmodels` wymagają czystego indeksu czasowego o stałej lub zdefiniowanej częstotliwości bez stref czasowych.
* **`calculate_returns(df, target_col, method)`**: Wyznacza stopy zwrotu.
  * *Jak działa:* Oblicza proste stopy zwrotu za pomocą `pct_change()` lub logarytmiczne za pomocą różnicy logarytmów naturalnych cen:
    $$R_t = \ln(P_t) - \ln(P_{t-1}) = \ln\left(\frac{P_t}{P_{t-1}}\right)$$
  * *Dlaczego tak:* Logarytmiczne stopy zwrotu są preferowane w badaniach ekonometrycznych ze względu na stabilność wariancji, ciągłą addytywność w czasie oraz wygładzenie rozkładu empirycznego w stronę rozkładu normalnego.
* **`search_symbols(query)`**: Wyszukuje instrumenty w API Yahoo Finance w czasie rzeczywistym.
  * *Jak działa:* Wysyła asynchroniczne żądanie HTTP GET do endpointu wyszukiwania Yahoo Finance:
    `https://query1.finance.yahoo.com/v1/finance/search?q={query}`
    Podmienia nagłówek `User-Agent` na wersję przeglądarkową Chrome.
  * *Dlaczego tak:* **Kluczowa poprawka stabilności:** Pierwotne zapytania kierowane na endpoint `query2` zwracały błąd HTTP 429 (Too Many Requests). Zastosowanie endpointu `query1` z kompletnym nagłówkiem User-Agent rozwiązało problem blokowania zapytań przez filtry antybotowe Yahoo Finance.

---

### B. Obliczenia Ekonometryczne (`src/analytics.py`)

Moduł implementuje matematyczne jądro aplikacji do analizy trendów, testów stacjonarności oraz estymacji modeli ARIMA/SARIMA.

#### Funkcje składowe:
* **`estimate_trend(df, target_col)`**: Wyznacza deterministyczny trend liniowy metodą najmniejszych kwadratów (OLS).
  * *Jak działa:* Buduje model regresji liniowej za pomocą `scikit-learn`:
    $$\hat{Y}_t = \beta_0 + \beta_1 \cdot t$$
    gdzie zmienną czasową $t$ jest indeks porządkowy $1, 2, \dots, T$. Funkcja zwraca współczynniki $\beta_0$ (intercept), $\beta_1$ (slope) oraz współczynnik determinacji $R^2$.
  * *Dlaczego tak:* Regresja OLS pozwala na oszacowanie stałego, liniowego przyrostu wartości aktywa w czasie i służy jako prosty estymator bazowy (baseline).
* **`check_stationarity(series)`**: Testuje stacjonarność szeregu czasowego.
  * *Jak działa:* Przeprowadza Rozszerzony Test Dickeya-Fullera (ADF) poprzez `statsmodels.tsa.stattools.adfuller`. Zwraca statystykę testową ADF, $p$-value, wartości krytyczne i automatyczną decyzję statystyczną na poziomie istotności $\alpha = 0.05$.
  * *Dlaczego tak:* Założenie stacjonarności jest fundamentalne dla modeli ARMA. Test ADF pozwala określić, czy szereg posiada trend stochastyczny (pierwiastek jednostkowy) i wymaga różnicowania ($d > 0$).
* **`fit_arima_model(series, order, seasonal_order)`**: Estymuje parametry modelu SARIMAX.
  * *Jak działa:* Wykorzystuje klasę `SARIMAX` z biblioteki `statsmodels` z wyłączonym wymuszaniem stacjonarności i odwracalności parametrów (`enforce_stationarity=False`, `enforce_invertibility=False`).
  * *Dlaczego tak:* W aplikacji interaktywnej użytkownik eksperymentuje z różnymi parametrami. Wyłączenie powyższych rygorów numerycznych zapobiega awariom procesu estymacji (brak zbieżności algorytmu maksymalnej wiarygodności BFGS).
* **`generate_forecast(model_result, steps)`**: Generuje prognozę punktową oraz przedziały ufności w przód.
  * *Jak działa:* Pobiera wynik modelu, pobiera ostatnią datę historyczną i generuje przyszły indeks roboczy za pomocą `pd.bdate_range(...)` (pomijając weekendy).
  * *Dlaczego tak:* Zapobiega to błędom logicznym polegającym na prognozowaniu notowań giełdowych na soboty i niedziele (giełdy są wtedy zamknięte).

---

### C. Warstwa Wizualizacji (`src/visualization.py`)

Odpowiada za budowanie wykresów w formacie wektorowym/interaktywnym.

#### Wykresy składowe:
* **`plot_price_and_trend`**: Wykres cenowy świecowy (OHLC) lub liniowy z dynamicznie nakładanym trendem liniowym OLS.
* **`plot_acf_pacf`**: Korelogramy autokorelacji i autokorelacji cząstkowej.
  * *Jak działa:* Rysuje współczynniki w formie słupkowej Plotly. Nakłada na wykres błękitny korytarz ufności 95% wyznaczony ze wzoru:
    $$\text{CI} = \pm \frac{1.96}{\sqrt{T}}$$
    Jeśli słupek wystaje poza korytarz, korelacja jest statystycznie istotna.
* **`plot_forecast`**: Wykres prognozy z płynnym połączeniem ostatniej ceny historycznej z prognozą oraz cieniowaniem przedziału ufności 95% (`fill='tonexty'`). Wyświetla tylko ostatnie 100 dni historii w celu zachowania optymalnej czytelności.

---

### D. Interfejs Użytkownika i Zarządzanie Stanem (`app.py`)

Aplikacja Streamlit koordynuje przepływ danych i zapewnia ergonomiczne środowisko pracy.
* **Zarządzanie stanem (Session State):** Streamlit przeładowuje cały skrypt przy każdej zmianie widgetu. Zastosowanie `st.session_state` dla kluczy `stock_data` oraz `model_fit_results` zapobiega ponownemu odpytywaniu API Yahoo przy zmianie parametrów wykresu lub modelu.
* **Dwustopniowa wyszukiwarka:** Użytkownik wpisuje tekst w pole wyszukiwania, system asynchronicznie odpytuje API, a użytkownik wybiera konkretny instrument z listy `st.selectbox`.

---

## 3. Metodologia Ekonometryczna (Teoria do Pracy)

Aplikacja w pełni realizuje klasyczną **metodologię Boxa-Jenkinsa** modelowania szeregów czasowych:

```
[KROK 1: Identyfikacja] -> [KROK 2: Estymacja] -> [KROK 3: Diagnostyka] -> [KROK 4: Prognozowanie]
        |                       |                      |
    Test ADF,               SARIMAX                Test Ljunga-Boxa,
   ACF/PACF               (Max. Wiarygodności)     Test Jarque-Bera
```

### Metryki diagnostyczne zaimplementowane w aplikacji:

#### 1. Kryteria Informacyjne (Wybór Modelu):
Kryteria te nakładają karę za nadmierną liczbę parametrów (zasada parsymonii modelu):
* **AIC (Akaike Information Criterion):**
  $$\text{AIC} = 2k - 2\ln(L)$$
* **BIC (Bayesian Information Criterion):**
  $$\text{BIC} = k\ln(T) - 2\ln(L)$$
  *gdzie $k$ to liczba parametrów, $L$ to funkcja wiarygodności, a $T$ to liczba obserwacji. Wybieramy model o najniższej wartości AIC/BIC.*

#### 2. Test Ljunga-Boxa (Autokorelacja residuów):
Służy do badania autokorelacji reszt modelu.
* $H_0$: Reszty modelu nie wykazują autokorelacji (są białym szumem).
* $H_1$: Reszty wykazują autokorelację (model nie wychwycił całej struktury zależności).
  *Jeśli p-value $> 0.05$, przyjmujemy $H_0$, co oznacza, że model jest poprawnie wyspecyfikowany.*

#### 3. Test Jarque-Bera (Normalność rozkładu residuów):
Bada skośność ($S$) i kurtozę ($K$) rozkładu reszt w stosunku do rozkładu normalnego ($S=0$, $K=3$).
* $H_0$: Reszty mają rozkład normalny.
* $H_1$: Reszty nie mają rozkładu normalnego.
  *W finansach p-value dla tego testu jest prawie zawsze mniejsze niż 0.05. Wynika to z faktu, że stopy zwrotu charakteryzują się grubymi ogonami i leptokurtozą, co warto opisać w analizie ekonometrycznej jako cechę empiryczną rynku.*
