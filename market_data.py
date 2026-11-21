
import yfinance as yf
import pandas as pd


def analyze_market(ticker):
    print(f"\nAnaliza instrumentu: {ticker}")

    try:
        stock = yf.Ticker(ticker)

        data = stock.history(
            period="6mo",
            interval="1d",
            auto_adjust=True
        )

        if data.empty:
            print("Brak dostępnych danych.")
            return

        close = data["Close"]

        data["SMA20"] = close.rolling(20).mean()
        data["SMA50"] = close.rolling(50).mean()

        latest = data.iloc[-1]

        print(f"Ostatnia cena: {latest['Close']:.2f}")
        print(f"Średnia SMA20: {latest['SMA20']:.2f}")
        print(f"Średnia SMA50: {latest['SMA50']:.2f}")

        if latest["SMA20"] > latest["SMA50"]:
            print("Trend: wzrostowy")
        else:
            print("Trend: spadkowy lub neutralny")

        print(f"Liczba sesji: {len(data)}")

    except Exception as error:
        print(f"Blad analizy: {error}")


if __name__ == "__main__":
    analyze_market("SPY")
    analyze_market("QQQ")
