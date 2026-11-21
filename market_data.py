
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

        # RSI 14
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

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        price = latest["Close"]
        sma20 = latest["SMA20"]
        sma50 = latest["SMA50"]
        rsi = latest["RSI"]

        change = ((price / previous["Close"]) - 1) * 100

        print(f"Cena: {price:.2f} PLN")
        print(f"Zmiana dzienna: {change:+.2f}%")
        print(f"SMA20: {sma20:.2f}")
        print(f"SMA50: {sma50:.2f}")
        print(f"RSI: {rsi:.2f}")

        # Ocena trendu
        if price > sma20 > sma50:
            trend = "WZROSTOWY"
        elif price < sma20 < sma50:
            trend = "SPADKOWY"
        else:
            trend = "BOCZNY / NIEJEDNOZNACZNY"

        print(f"\nTrend: {trend}")

        # Ocena RSI
        if rsi < 30:
            print("RSI: Instrument moze byc wyprzedany.")
        elif rsi > 70:
            print("RSI: Instrument moze byc wykupiony.")
        else:
            print("RSI: Neutralny zakres.")

        # Prosta ocena sygnalu
        score = 0

        if price > sma20:
            score += 1

        if sma20 > sma50:
            score += 1

        if rsi < 70:
            score += 1

        if score == 3:
            signal = "POZYTYWNY"
        elif score == 0:
            signal = "NEGATYWNY"
        else:
            signal = "NEUTRALNY"

        print(f"\nOcena MARKET AI: {signal}")
        print(f"Wynik analizy: {score}/3")
        print(f"Liczba sesji: {len(data)}")

        return {
            "ticker": ticker,
            "price": round(float(price), 2),
            "change": round(float(change), 2),
            "rsi": round(float(rsi), 2),
            "trend": trend,
            "signal": signal,
            "score": score
        }

    except Exception as error:
        print(f"Blad analizy: {error}")
        return None
