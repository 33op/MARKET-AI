
import math
import yfinance as yf
import pandas as pd


VERSION = "15.0"

HISTORY_PERIOD = "2y"
MIN_SESSIONS = 60

# Koszty symulowanej transakcji
COMMISSION_RATE = 0.003
SLIPPAGE_RATE = 0.001


def safe_float(value):
    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return None


def download_history(ticker):
    try:
        data = yf.Ticker(ticker).history(
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=True
        )

        if data is None or data.empty:
            return None

        if "Close" not in data.columns:
            return None

        data = data.copy()
        data = data.sort_index()
        data = data[~data.index.duplicated(keep="last")]

        data["Close"] = pd.to_numeric(
            data["Close"],
            errors="coerce"
        )

        data = data.dropna(subset=["Close"])
        data = data[data["Close"] > 0]

        if "Volume" not in data.columns:
            data["Volume"] = 0

        data["Volume"] = pd.to_numeric(
            data["Volume"],
            errors="coerce"
        )

        if data["Volume"].dropna().lt(0).any():
            return None

        data["Volume"] = data["Volume"].fillna(0)

        if not data["Volume"].map(
            lambda value: math.isfinite(float(value))
        ).all():
            return None

        if len(data) < MIN_SESSIONS:
            return None

        return data

    except Exception as error:
        print(f"BACKTEST: {ticker}: {error}")
        return None


def calculate_indicators(data):
    """
    Wskaźniki obliczane tak samo jak w market_data.py.
    """

    data = data.copy()

    close = data["Close"]
    volume = data["Volume"]

    # SMA
    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        min_periods=14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))

    data["RSI"] = 100 - (100 / (1 + rs))

    data.loc[
        (avg_loss == 0) & (avg_gain > 0),
        "RSI"
    ] = 100

    data.loc[
        (avg_loss == 0) & (avg_gain == 0),
        "RSI"
    ] = 50

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = ema12 - ema26

    data["SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    # Wolumen
    data["AVG_VOLUME"] = (
        volume.shift(1).rolling(20).mean()
    )

    # Momentum
    data["MOMENTUM_5"] = (
        close.pct_change(5) * 100
    )

    data["MOMENTUM_20"] = (
        close.pct_change(20) * 100
    )

    # Zmiana dzienna
    data["DAILY_CHANGE"] = (
        close.pct_change() * 100
    )

    # Zmienność
    data["VOLATILITY"] = (
        close.pct_change()
        .rolling(20)
        .std()
        * math.sqrt(252)
        * 100
    )

    # Rosnąca SMA20
    data["SMA20_RISING"] = (
        data["SMA20"] > data["SMA20"].shift(5)
    )

    # Potwierdzenie sygnału
    data["CONFIRMATION"] = (
        (data["Close"] > data["SMA20"])
        & (data["SMA20"] > data["SMA50"])
        & (data["MACD"] > data["SIGNAL"])
    )

    data["CONFIRMED_DAYS"] = (
        data["CONFIRMATION"]
        .rolling(3)
        .sum()
    )

    return data


def calculate_signal(row):
    """
    Punktacja zgodna z aktualnym market_data.py.
    """

    required_values = [
        row["Close"],
        row["SMA20"],
        row["SMA50"],
        row["RSI"],
        row["MACD"],
        row["SIGNAL"],
        row["MOMENTUM_5"],
        row["MOMENTUM_20"],
        row["DAILY_CHANGE"],
        row["VOLATILITY"],
        row["CONFIRMED_DAYS"]
    ]

    if any(pd.isna(value) for value in required_values):
        return None

    price = safe_float(row["Close"])
    sma20 = safe_float(row["SMA20"])
    sma50 = safe_float(row["SMA50"])
    rsi = safe_float(row["RSI"])
    macd = safe_float(row["MACD"])
    macd_signal = safe_float(row["SIGNAL"])

    momentum_5d = safe_float(row["MOMENTUM_5"])
    momentum_20d = safe_float(row["MOMENTUM_20"])

    change = safe_float(row["DAILY_CHANGE"])
    volatility = safe_float(row["VOLATILITY"])

    confirmed_days = safe_float(
        row["CONFIRMED_DAYS"]
    )

    values = [
        price,
        sma20,
        sma50,
        rsi,
        macd,
        macd_signal,
        momentum_5d,
        momentum_20d,
        change,
        volatility,
        confirmed_days
    ]

    if any(value is None for value in values):
        return None

    if price <= 0 or sma20 <= 0 or sma50 <= 0:
        return None

    if not 0 <= rsi <= 100:
        return None

    confirmed_days = int(confirmed_days)

    # Wolumen
    avg_volume = safe_float(row["AVG_VOLUME"])
    current_volume = safe_float(row["Volume"])

    volume_ratio = None

    if (
        avg_volume is not None
        and avg_volume > 0
        and current_volume is not None
    ):
        volume_ratio = current_volume / avg_volume

        if not math.isfinite(volume_ratio):
            volume_ratio = None

    # OCENA BAZOWA
    raw_score = 0

    if price > sma20:
        raw_score += 1

    if sma20 > sma50:
        raw_score += 1

    if bool(row["SMA20_RISING"]):
        raw_score += 1

    if macd > macd_signal:
        raw_score += 2

    if 50 <= rsi <= 65:
        raw_score += 2

    elif 40 <= rsi < 50 or 65 < rsi <= 70:
        raw_score += 1

    if momentum_5d > 0:
        raw_score += 1

    if momentum_20d > 0:
        raw_score += 1

    if (
        volume_ratio is not None
        and volume_ratio >= 1.2
        and change > 0
    ):
        raw_score += 1

    # KARY
    penalty = 0

    if rsi > 80:
        penalty += 2

    elif rsi > 70:
        penalty += 1

    if change >= 10:
        penalty += 2

    elif change >= 5:
        penalty += 1

    if (
        volume_ratio is not None
        and volume_ratio >= 4
    ):
        penalty += 1

    if volatility >= 60:
        penalty += 2

    elif volatility >= 40:
        penalty += 1

    score = max(
        0,
        min(10, raw_score - penalty)
    )

    # SYGNAŁ
    if score >= 7 and confirmed_days >= 2:
        signal = "POZYTYWNY"

    elif score >= 4:
        signal = "NEUTRALNY"

    else:
        signal = "NEGATYWNY"

    return {
        "score": score,
        "raw_score": raw_score,
        "penalty": penalty,
        "signal": signal,
        "price": price
    }


def calculate_trade_return(entry_price, exit_price):
    """
    Oblicza wynik po kosztach:
    - poślizg przy zakupie,
    - poślizg przy sprzedaży,
    - prowizja przy zakupie,
    - prowizja przy sprzedaży.
    """

    if entry_price <= 0 or exit_price <= 0:
        return None

    actual_entry = (
        entry_price * (1 + SLIPPAGE_RATE)
    )

    actual_exit = (
        exit_price * (1 - SLIPPAGE_RATE)
    )

    buy_cost = (
        actual_entry * (1 + COMMISSION_RATE)
    )

    sell_value = (
        actual_exit * (1 - COMMISSION_RATE)
    )

    if buy_cost <= 0:
        return None

    result = (
        sell_value / buy_cost - 1
    ) * 100

    if not math.isfinite(result):
        return None

    return result


def backtest_ticker(ticker):
    data = download_history(ticker)

    if data is None:
        return None

    data = calculate_indicators(data)

    signals = []

    # Sygnał powstaje po zamknięciu sesji.
    # Wejście: zamknięcie kolejnej sesji.
    # Wyjście: zamknięcie następnej sesji.
    #
    # Dzięki temu nie wykorzystujemy ceny,
    # po której nie mogliśmy jeszcze zareagować.

    for index in range(60, len(data) - 2):

        row = data.iloc[index]

        result = calculate_signal(row)

        if result is None:
            continue

        entry_price = safe_float(
            data["Close"].iloc[index + 1]
        )

        exit_price = safe_float(
            data["Close"].iloc[index + 2]
        )

        if entry_price is None or exit_price is None:
            continue

        trade_return = calculate_trade_return(
            entry_price,
            exit_price
        )

        if trade_return is None:
            continue

        signals.append({
            "ticker": ticker,
            "date": str(data.index[index].date()),
            "entry_date": str(
                data.index[index + 1].date()
            ),
            "exit_date": str(
                data.index[index + 2].date()
            ),
            "signal": result["signal"],
            "score": result["score"],
            "raw_score": result["raw_score"],
            "penalty": result["penalty"],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return": trade_return
        })

    if not signals:
        return None

    positive = [
        item for item in signals
        if item["signal"] == "POZYTYWNY"
    ]

    if not positive:
        return {
            "ticker": ticker,
            "signals": len(signals),
            "positive_count": 0,
            "average_return": None,
            "median_return": None,
            "win_rate": None,
            "costs_included": True
        }

    returns = [
        item["return"]
        for item in positive
    ]

    average_return = sum(returns) / len(returns)

    median_return = float(
        pd.Series(returns).median()
    )

    win_rate = (
        sum(value > 0 for value in returns)
        / len(returns)
        * 100
    )

    return {
        "ticker": ticker,
        "signals": len(signals),
        "positive_count": len(positive),
        "average_return": round(
            average_return,
            2
        ),
        "median_return": round(
            median_return,
            2
        ),
        "win_rate": round(
            win_rate,
            1
        ),
        "costs_included": True
    }


def run_backtest(instruments):
    print("=" * 65)
    print(f"MARKET AI {VERSION} - BACKTEST")
    print("=" * 65)

    print("Historia:", HISTORY_PERIOD)
    print("Punktacja: zgodna z market_data.py")
    print("Wejście: zamknięcie kolejnej sesji")
    print("Wyjście: zamknięcie następnej sesji")
    print(
        f"Prowizja: {COMMISSION_RATE * 100:.2f}% na stronę"
    )
    print(
        f"Poślizg: {SLIPPAGE_RATE * 100:.2f}% na stronę"
    )

    results = []

    for ticker in instruments:
        print(f"Testowanie: {ticker}")

        result = backtest_ticker(ticker)

        if result is not None:
            results.append(result)

    results.sort(
        key=lambda item: (
            item["average_return"]
            if item["average_return"] is not None
            else -999
        ),
        reverse=True
    )

    print("\nWYNIKI BACKTESTU")
    print("=" * 65)

    for item in results:
        print(
            f"{item['ticker']} | "
            f"Sygnały: {item['signals']} | "
            f"Pozytywne: {item['positive_count']} | "
            f"Średni wynik: {item['average_return']}% | "
            f"Mediana: {item['median_return']}% | "
            f"Skuteczność: {item['win_rate']}%"
        )

    print("=" * 65)

    total_positive = sum(
        item["positive_count"]
        for item in results
    )

    print(
        f"Liczba pozytywnych sygnałów: {total_positive}"
    )

    print(
        "UWAGA: Wyniki dotyczą pojedynczych sygnałów, "
        "a nie pełnego portfela."
    )

    print(
        "Sygnały mogą się nakładać, więc nie są "
        "niezależnymi transakcjami."
    )

    print("=" * 65)

    return results
