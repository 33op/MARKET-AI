
import math
import yfinance as yf
import pandas as pd


def analyze_market(ticker):
    print("\n" + "=" * 45)
    print(f"ANALIZA: {ticker}")
    print("=" * 45)

    try:
        stock = yf.Ticker(ticker)

        data = stock.history(
            period="1y",
            interval="1d",
            auto_adjust=True
        )

        if data.empty or len(data) < 60:
            print("Brak wystarczajacych danych.")
            return None

        if "Volume" not in data.columns:
            data["Volume"] = 0

        data = data.dropna(subset=["Close"])

        close = data["Close"]
        volume = data["Volume"]

        # SREDNIE KROCZACE
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
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        data["MACD"] = ema12 - ema26

        data["SIGNAL"] = data["MACD"].ewm(
            span=9,
            adjust=False
        ).mean()

        # WOLUMEN
        data["AVG_VOLUME"] = volume.shift(1).rolling(20).mean()

        # DANE Z OSTATNIEJ SESJI
        latest = data.iloc[-1]
        previous = data.iloc[-2]

        price = float(latest["Close"])
        previous_price = float(previous["Close"])

        sma20 = float(latest["SMA20"])
        sma50 = float(latest["SMA50"])

        rsi = float(latest["RSI"])

        macd = float(latest["MACD"])
        macd_signal = float(latest["SIGNAL"])

        change = (
            price / previous_price - 1
        ) * 100

        # MOMENTUM 5 I 20 DNI
        momentum_5d = (
            price / float(close.iloc[-6]) - 1
        ) * 100

        momentum_20d = (
            price / float(close.iloc[-21]) - 1
        ) * 100

        # ZMIENNOSC ROCZNA
        daily_returns = close.pct_change()

        volatility = (
            daily_returns.tail(20).std()
            * math.sqrt(252)
            * 100
        )

        # POROWNANIE WOLUMENU
        avg_volume = float(latest["AVG_VOLUME"])
        current_volume = float(latest["Volume"])

        volume_ratio = None

        if pd.notna(avg_volume) and avg_volume > 0:
            volume_ratio = current_volume / avg_volume

        # TREND
        if price > sma20 > sma50:
            trend = "WZROSTOWY"
        elif price < sma20 < sma50:
            trend = "SPADKOWY"
        else:
            trend = "BOCZNY"

        # POTWIERDZENIE SYGNALU
        confirmation = (
            (data["Close"] > data["SMA20"])
            & (data["SMA20"] > data["SMA50"])
            & (data["MACD"] > data["SIGNAL"])
        )

        confirmed_days = int(
            confirmation.tail(3).sum()
        )

        # SYSTEM PUNKTOWY
        score = 0

        if price > sma20:
            score += 1

        if sma20 > sma50:
            score += 1

        if float(data["SMA20"].iloc[-1]) > float(
            data["SMA20"].iloc[-6]
        ):
            score += 1

        if macd > macd_signal:
            score += 2

        if 50 <= rsi <= 65:
            score += 2
        elif 40 <= rsi < 50 or 65 < rsi <= 70:
            score += 1

        if momentum_5d > 0:
            score += 1

        if momentum_20d > 0:
            score += 1

        if (
            volume_ratio is not None
            and volume_ratio >= 1.2
            and change > 0
        ):
            score += 1

        # SYGNAL Z POTWIERDZENIEM
        if score >= 7 and confirmed_days >= 2:
            signal = "POZYTYWNY"
        elif score >= 4 or (
            score >= 7 and confirmed_days < 2
        ):
            signal = "NEUTRALNY"
        else:
            signal = "NEGATYWNY"

        # OCENA RYZYKA
        if (
            volatility >= 60
            or change <= -5
            or rsi < 25
        ):
            risk = "WYSOKIE"

        elif (
            volatility >= 40
            or change <= -3
            or rsi > 75
        ):
            risk = "PODWYZSZONE"

        else:
            risk = "UMIARKOWANE"

        # OSTRZEZENIA
        warnings = []

        if change <= -4:
            warnings.append("MOCNY SPADEK DZIENNY")

        if change >= 4:
            warnings.append("MOCNY WZROST DZIENNY")

        if rsi < 30:
            warnings.append("RSI: OBSZAR WYPRZEDANIA")

        if rsi > 70:
            warnings.append("RSI: OBSZAR WYKUPIENIA")

        if volatility >= 60:
            warnings.append("BARDZO WYSOKA ZMIENNOSC")

        if (
            volume_ratio is not None
            and volume_ratio >= 1.5
        ):
            warnings.append("WYSOKI WOLUMEN")

        if confirmed_days < 2:
            warnings.append("SYGNAL NIE MA POTWIERDZENIA")

        # WYNIKI
        print(f"Cena: {price:.2f} PLN")
        print(f"Zmiana dzienna: {change:+.2f}%")
        print(f"SMA20: {sma20:.2f}")
        print(f"SMA50: {sma50:.2f}")
        print(f"RSI: {rsi:.2f}")
        print(f"MACD: {macd:.4f}")
        print(f"MACD Signal: {macd_signal:.4f}")

        if volume_ratio is not None:
            print(f"Wolumen: {volume_ratio:.2f}x sredniej")
        else:
            print("Wolumen: brak danych")

        print(f"Momentum 5 dni: {momentum_5d:+.2f}%")
        print(f"Momentum 20 dni: {momentum_20d:+.2f}%")
        print(f"Zmiennosc roczna: {volatility:.2f}%")
        print(f"Potwierdzenie: {confirmed_days}/3 sesji")

        print(f"\nTrend: {trend}")
        print(f"Ocena: {signal}")
        print(f"Wynik: {score}/10")
        print(f"Ryzyko: {risk}")

        if warnings:
            print("\nOSTRZEZENIA:")
            for warning in warnings:
                print(f"- {warning}")
        else:
            print("\nBrak ostrzezen.")

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "change": round(change, 2),
            "rsi": round(rsi, 2),
            "trend": trend,
            "signal": signal,
            "score": score,
            "risk": risk,
            "macd": round(macd, 4),
            "volume_ratio": (
                round(volume_ratio, 2)
                if volume_ratio is not None else None
            ),
            "volatility": round(volatility, 2),
            "confirmed_days": confirmed_days,
            "momentum_5d": round(momentum_5d, 2),
            "momentum_20d": round(momentum_20d, 2)
        }

    except Exception as error:
        print(f"BLAD ANALIZY {ticker}: {error}")
        return None
