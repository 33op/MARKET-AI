
import math
import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf


VERSION = "16.6"

HISTORY_PERIOD = "1y"
INTERVAL = "1d"
MIN_SESSIONS = 60
MAX_DATA_AGE_DAYS = 7


def safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def calculate_rsi(close, period=14):
    delta = close.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.mask(avg_loss == 0, 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)

    return rsi


def calculate_indicators(data):
    data = data.copy()

    data["SMA20"] = data["Close"].rolling(20).mean()
    data["SMA50"] = data["Close"].rolling(50).mean()

    data["RSI"] = calculate_rsi(data["Close"])

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["VOLUME_AVG"] = data["Volume"].rolling(20).mean()

    data["MOMENTUM_5"] = data["Close"].pct_change(5) * 100
    data["MOMENTUM_20"] = data["Close"].pct_change(20) * 100

    daily_returns = data["Close"].pct_change()

    data["VOLATILITY"] = (
        daily_returns.rolling(20).std() * math.sqrt(252) * 100
    )

    return data


def analyze_market(ticker):
    warnings_list = []
    strengths = []
    risk_factors = []
    penalties = []

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            data = yf.Ticker(ticker).history(
                period=HISTORY_PERIOD,
                interval=INTERVAL,
                auto_adjust=True
            )

        if data is None or data.empty:
            return None

        required = {"Close", "Volume"}

        if not required.issubset(data.columns):
            return None

        data = data.copy().sort_index()
        data = data[~data.index.duplicated(keep="last")]

        if data.index.tz is not None:
            data.index = data.index.tz_localize(None)

        data["Close"] = pd.to_numeric(
            data["Close"], errors="coerce"
        )

        data["Volume"] = pd.to_numeric(
            data["Volume"], errors="coerce"
        )

        data = data.dropna(subset=["Close", "Volume"])

        data = data[
            (data["Close"] > 0)
            & (data["Close"].map(math.isfinite))
        ]

        if len(data) < MIN_SESSIONS:
            return None

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        latest_date = pd.Timestamp(data.index[-1]).date()
        today = datetime.now().date()

        data_age_days = max(
            0,
            (today - latest_date).days
        )

        if data_age_days > MAX_DATA_AGE_DAYS:
            warnings_list.append(
                f"Dane są nieaktualne: {data_age_days} dni"
            )

        data = calculate_indicators(data)

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        price = safe_float(latest["Close"])

        if price <= 0:
            return None

        sma20 = safe_float(latest["SMA20"])
        sma50 = safe_float(latest["SMA50"])
        rsi = safe_float(latest["RSI"], 50)

        macd = safe_float(latest["MACD"])
        macd_signal = safe_float(latest["MACD_SIGNAL"])

        volume = safe_float(latest["Volume"])
        volume_avg = safe_float(latest["VOLUME_AVG"])

        momentum_5 = safe_float(latest["MOMENTUM_5"])
        momentum_20 = safe_float(latest["MOMENTUM_20"])

        volatility = safe_float(latest["VOLATILITY"])

        previous_close = safe_float(previous["Close"])

        daily_change = (
            (price / previous_close - 1) * 100
            if previous_close > 0 else 0
        )

        volume_ratio = (
            volume / volume_avg
            if volume_avg > 0 else 0
        )

        raw_score = 0

        # TREND
        if price > sma20 > sma50:
            raw_score += 2
            strengths.append("Silny trend wzrostowy")
        elif price > sma20:
            raw_score += 1
            strengths.append("Cena powyzej SMA20")
        elif price < sma20 < sma50:
            raw_score -= 2
            risk_factors.append("Silny trend spadkowy")
        else:
            risk_factors.append("Slaby trend")

        # RSI
        if 50 <= rsi <= 65:
            raw_score += 2
            strengths.append("Korzystny poziom RSI")
        elif 40 <= rsi < 50:
            raw_score += 1
        elif rsi > 75:
            raw_score -= 2
            risk_factors.append("Wykupienie rynku")
        elif rsi < 30:
            raw_score -= 1
            risk_factors.append("Mozliwe wyprzedanie")
        elif rsi < 40:
            raw_score -= 1

        # MACD
        if macd > macd_signal:
            raw_score += 2
            strengths.append("MACD powyzej linii sygnalowej")
        else:
            raw_score -= 1
            risk_factors.append("Negatywny MACD")

        # MOMENTUM
        if momentum_5 > 0 and momentum_20 > 0:
            raw_score += 2
            strengths.append("Dodatnie momentum")
        elif momentum_5 < 0 and momentum_20 < 0:
            raw_score -= 2
            risk_factors.append("Ujemne momentum")
        elif momentum_20 > 0:
            raw_score += 1

        # WOLUMEN
        if volume_ratio >= 1.5:
            raw_score += 1
            strengths.append("Podwyzszony wolumen")
        elif 0 < volume_ratio < 0.5:
            risk_factors.append("Niski wolumen")

        # RYZYKO
        score_penalty = 0

        if volatility > 60:
            score_penalty += 2
            penalties.append("Bardzo wysoka zmiennosc")
        elif volatility > 40:
            score_penalty += 1
            penalties.append("Wysoka zmiennosc")

        if rsi > 80:
            score_penalty += 1
            penalties.append("Ekstremalne wykupienie")

        if data_age_days > 3:
            score_penalty += 1
            penalties.append("Starsze dane rynkowe")

        score = max(0, min(10, raw_score - score_penalty + 5))

        # POTWIERDZENIE SYGNALU
        confirmation_conditions = [
            price > sma20,
            sma20 > sma50,
            macd > macd_signal
        ]

        confirmed_days = sum(confirmation_conditions)

        if score >= 7 and confirmed_days >= 2:
            signal = "POZYTYWNY"
        elif score >= 4:
            signal = "NEUTRALNY"
        else:
            signal = "NEGATYWNY"

        # RYZYKO
        if volatility >= 60 or score_penalty >= 3:
            risk = "WYSOKIE"
        elif volatility >= 35 or score_penalty >= 2:
            risk = "UMIARKOWANE"
        else:
            risk = "NISKIE"

        if data_age_days > MAX_DATA_AGE_DAYS:
            signal = "NEUTRALNY"
            warnings_list.append(
                "Wymuszono sygnal neutralny z powodu starych danych"
            )

        return {
            "ticker": ticker,
            "date": str(latest_date),
            "price": round(price, 4),
            "change": round(daily_change, 2),
            "daily_change": round(daily_change, 2),

            "RSI": round(rsi, 2),
            "trend": (
                "WZROSTOWY" if price > sma20 > sma50
                else "SPADKOWY" if price < sma20 < sma50
                else "BOCZNY"
            ),

            "signal": signal,
            "score": int(score),
            "raw_score": int(raw_score),
            "score_penalty": int(score_penalty),
            "risk": risk,

            "MACD": round(macd, 6),
            "volume_ratio": round(volume_ratio, 2),
            "volatility": round(volatility, 2),

            "confirmed_days": int(confirmed_days),
            "confirmation_days": int(confirmed_days),

            "momentum": {
                "5d": round(momentum_5, 2),
                "20d": round(momentum_20, 2)
            },

            "data_status": (
                "AKTUALNE"
                if data_age_days <= MAX_DATA_AGE_DAYS
                else "NIEAKTUALNE"
            ),

            "data_age_days": int(data_age_days),
            "sessions_count": int(len(data)),

            "strengths": strengths,
            "risk_factors": risk_factors,
            "warnings": warnings_list,
            "penalties": penalties
        }

    except Exception as error:
        print(f"Blad analizy {ticker}: {error}")
        return None
