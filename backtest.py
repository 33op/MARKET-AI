
import math
import yfinance as yf
import pandas as pd


VERSION = "14.0"
HISTORY_PERIOD = "2y"
MIN_SESSIONS = 60

INITIAL_CAPITAL = 20000.0
COMMISSION_RATE = 0.003
SLIPPAGE_RATE = 0.001
MAX_POSITIONS = 3


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

        data = data.copy().sort_index()
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
        ).fillna(0)

        if len(data) < MIN_SESSIONS:
            return None

        return data

    except Exception as error:
        print(f"BACKTEST: {ticker}: {error}")
        return None


def calculate_indicators(data):
    data = data.copy()

    close = data["Close"]
    volume = data["Volume"]

    data["SMA20"] = close.rolling(20).mean()
    data["SMA50"] = close.rolling(50).mean()

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

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26

    data["SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["AVG_VOLUME"] = volume.shift(1).rolling(20).mean()

    data["MOMENTUM_5"] = close.pct_change(5) * 100
    data["MOMENTUM_20"] = close.pct_change(20) * 100

    data["DAILY_CHANGE"] = close.pct_change() * 100

    data["VOLATILITY"] = (
        close.pct_change()
        .rolling(20)
        .std()
        * math.sqrt(252)
        * 100
    )

    data["SMA20_RISING"] = data["SMA20"] > data["SMA20"].shift(5)

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
    values = [
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

    if any(pd.isna(value) for value in values):
        return None

    price = float(row["Close"])
    sma20 = float(row["SMA20"])
    sma50 = float(row["SMA50"])
    rsi = float(row["RSI"])
    macd = float(row["MACD"])
    macd_signal = float(row["SIGNAL"])
    momentum_5d = float(row["MOMENTUM_5"])
    momentum_20d = float(row["MOMENTUM_20"])
    change = float(row["DAILY_CHANGE"])
    volatility = float(row["VOLATILITY"])
    confirmed_days = int(row["CONFIRMED_DAYS"])

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

    avg_volume = safe_float(row["AVG_VOLUME"])
    current_volume = safe_float(row["Volume"])

    volume_ratio = None

    if (
        avg_volume is not None
        and avg_volume > 0
        and current_volume is not None
    ):
        volume_ratio = current_volume / avg_volume

    if (
        volume_ratio is not None
        and volume_ratio >= 1.2
        and change > 0
    ):
        raw_score += 1

    penalty = 0

    if rsi > 80:
        penalty += 2
    elif rsi > 70:
        penalty += 1

    if change >= 10:
        penalty += 2
    elif change >= 5:
        penalty += 1

    if volume_ratio is not None and volume_ratio >= 4:
        penalty += 1

    if volatility >= 60:
        penalty += 2
    elif volatility >= 40:
        penalty += 1

    score = max(0, min(10, raw_score - penalty))

    if score >= 7 and confirmed_days >= 2:
        signal = "POZYTYWNY"
    elif score >= 4:
        signal = "NEUTRALNY"
    else:
        signal = "NEGATYWNY"

    return {
        "score": score,
        "signal": signal,
        "price": price
    }


def backtest_ticker(ticker):
    data = download_history(ticker)

    if data is None:
        return None

    data = calculate_indicators(data)

    signals = []

    for index in range(60, len(data) - 1):
        row = data.iloc[index]
        result = calculate_signal(row)

        if result is None:
            continue

        next_price = safe_float(data["Close"].iloc[index + 1])

        if next_price is None:
            continue

        signals.append({
            "date": str(data.index[index].date()),
            "next_date": str(data.index[index + 1].date()),
            "price": result["price"],
            "next_price": next_price,
            "score": result["score"],
            "signal": result["signal"],
            "return": (
                next_price / result["price"] - 1
            ) * 100
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
            "win_rate": None
        }

    returns = [item["return"] for item in positive]

    return {
        "ticker": ticker,
        "signals": len(signals),
        "positive_count": len(positive),
        "average_return": round(
            sum(returns) / len(returns),
            2
        ),
        "win_rate": round(
            sum(value > 0 for value in returns)
            / len(returns) * 100,
            1
        )
    }


def run_backtest(instruments):
    print("=" * 65)
    print(f"MARKET AI {VERSION} - BACKTEST")
    print("=" * 65)

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
            f"Skuteczność: {item['win_rate']}%"
        )

    print("=" * 65)

    return results
