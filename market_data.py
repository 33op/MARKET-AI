
import yfinance as yf
import pandas as pd


def analyze_market(ticker):
    print(f"\n{'=' * 45}")
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
            print("Za malo danych do analizy.")
            return None

        close = data["Close"]

        # Srednie kroczace
        data["SMA20"] = close.rolling(20).mean()
        data["SMA50"] = close.rolling(50).mean()

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(
            alpha=1/14, adjust=False
        ).mean()

        avg_loss = loss.ewm(
            alpha=1/14, adjust=False
        ).mean()

        rs = avg_gain / avg_loss.replace(0, float("nan"))
        data["RSI"] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        data["MACD"] = ema12 - ema26
        data["MACD_SIGNAL"] = data["MACD"].ewm(
            span=9, adjust=False
        ).mean()

        # Wolumen
        data["VOLUME_AVG20"] = data["Volume"].rolling(20).mean()

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        price = float(latest["Close"])
        sma20 = float(latest["SMA20"])
        sma50 = float(latest["SMA50"])
        rsi = float(latest["RSI"])
        macd = float(latest["MACD"])
        macd_signal = float(latest["MACD_SIGNAL"])

        change = (
            price / float(previous["Close"]) - 1
        ) * 100

        volume_avg = float(latest["VOLUME_AVG20"])
        volume = float(latest["Volume"])

        if volume_avg > 0:
            volume_ratio = volume / volume_avg
        else:
            volume_ratio = None

        # Trend
        if price > sma20 > sma50:
            trend = "WZROSTOWY"
        elif price < sma20 < sma50:
            trend = "SPADKOWY"
        else:
            trend = "BOCZNY / NIEJEDNOZNACZNY"

        # Punktacja: maksymalnie 10 punktow
        score = 0

        # Trend: 0-2
        if price > sma20 and sma20 > sma50:
            score += 2
        elif price > sma20 or sma20 > sma50:
            score += 1

        # MACD: 0-2
        if macd > macd_signal and macd > 0:
            score += 2
        elif macd > macd_signal:
            score += 1

        # RSI: 0-2
        if 50 <= rsi <= 65:
            score += 2
        elif 40 <= rsi < 50 or 65 < rsi <= 70:
            score += 1

        # Wolumen: 0-2
        if volume_ratio is not None:
            if volume_ratio >= 1.5 and change > 0:
                score += 2
            elif volume_ratio >= 1.2:
                score += 1

        # Zmiana dzienna: 0-2
        if change >= 1:
            score += 2
        elif change > 0:
            score += 1

        # Ocena
        if score >= 7:
            signal = "POZYTYWNY"
        elif score >= 4:
            signal = "NEUTRALNY"
        else:
            signal = "NEGATYWNY"

        # Raport
        print(f"Cena: {price:.2f} PLN")
        print(f"Zmiana dzienna: {change:+.2f}%")
        print(f"SMA20: {sma20:.2f}")
        print(f"SMA50: {sma50:.2f}")
        print(f"RSI: {rsi:.2f}")
        print(f"MACD: {macd:.4f}")
        print(f"MACD Signal: {macd_signal:.4f}")

        if volume_ratio is not None:
            print(f"Wolumen / srednia 20 sesji: {volume_ratio:.2f}x")
        else:
            print("Wolumen: brak danych")

        print(f"\nTrend: {trend}")
        print(f"Ocena MARKET AI: {signal}")
        print(f"Wynik: {score}/10")

        # Ostrzezenia
        warnings = []

        if change <= -4:
            warnings.append("MOCNY SPADEK DZIENNY")

        if rsi < 30:
            warnings.append("RSI W OBSZARZE WYPRZEDANIA")

        if volume_ratio is not None and volume_ratio >= 1.5 and change < 0:
            warnings.append("WYSOKI WOLUMEN PODCZAS SPADKU")

        if warnings:
            print("\nOSTRZEZENIA:")
            for warning in warnings:
                print(f"- {warning}")
        else:
            print("\nBrak wykrytych ostrzezen.")

        print(f"Liczba sesji: {len(data)}")

        return {
            "ticker": ticker,
            "price": round(price, 2),
            "change": round(change, 2),
            "rsi": round(rsi, 2),
            "trend": trend,
            "signal": signal,
            "score": score,
            "macd": round(macd, 4),
            "volume_ratio": (
                round(volume_ratio, 2)
                if volume_ratio is not None else None
            )
        }

    except Exception as error:
        print(f"Blad analizy: {error}")
        return None
