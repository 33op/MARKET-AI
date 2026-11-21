
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

        if data.empty or len(data) < 50:
            print("Brak wystarczajacych danych.")
            return None

        close = data["Close"]

        data["SMA20"] = close.rolling(20).mean()
        data["SMA50"] = close.rolling(50).mean()

        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(0, float("nan"))
        data["RSI"] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        data["MACD"] = ema12 - ema26
        data["SIGNAL"] = data["MACD"].ewm(
            span=9,
            adjust=False
        ).mean()

        data["AVG_VOLUME"] = data["Volume"].rolling(20).mean()

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        price = float(latest["Close"])
        sma20 = float(latest["SMA20"])
        sma50 = float(latest["SMA50"])
        rsi = float(latest["RSI"])
        macd = float(latest["MACD"])
        macd_signal = float(latest["SIGNAL"])

        change = (
            price / float(previous["Close"]) - 1
        ) * 100

        avg_volume = float(latest["AVG_VOLUME"])
        volume = float(latest["Volume"])

        volume_ratio = None

        if pd.notna(avg_volume) and avg_volume > 0:
            volume_ratio = volume / avg_volume

        # TREND
        if price > sma20 > sma50:
            trend = "WZROSTOWY"
        elif price < sma20 < sma50:
            trend = "SPADKOWY"
        else:
            trend = "BOCZNY"

        # SCORE
        score = 0

        if price > sma20:
            score += 1

        if sma20 > sma50:
            score += 1

        if macd > macd_signal:
            score += 2

        if 50 <= rsi <= 65:
            score += 2
        elif 40 <= rsi < 50 or 65 < rsi <= 70:
            score += 1

        if change > 0:
            score += 1

        if volume_ratio is not None and volume_ratio >= 1.2:
            score += 1

        if change >= 1:
            score += 2

        # SIGNAL
        if score >= 7:
            signal = "POZYTYWNY"
        elif score >= 4:
            signal = "NEUTRALNY"
        else:
            signal = "NEGATYWNY"

        # RISK
        if change <= -5 or rsi < 25:
            risk = "WYSOKIE"
        elif change <= -3 or rsi > 75:
            risk = "PODWYZSZONE"
        else:
            risk = "UMIARKOWANE"

        # WARNINGS
        warnings = []

        if change <= -4:
            warnings.append("MOCNY SPADEK DZIENNY")

        if change >= 4:
            warnings.append("MOCNY WZROST DZIENNY")

        if rsi < 30:
            warnings.append("RSI: OBSZAR WYPRZEDANIA")

        if rsi > 70:
            warnings.append("RSI: OBSZAR WYKUPIENIA")

        if volume_ratio is not None and volume_ratio >= 1.5:
            warnings.append("WYSOKI WOLUMEN")

        # RESULTS
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
            )
        }

    except Exception as error:
        print(f"BLAD ANALIZY {ticker}: {error}")
        return None
