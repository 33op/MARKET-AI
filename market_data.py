
import math
import yfinance as yf
import pandas as pd


MAX_DATA_AGE_DAYS = 7
MIN_SESSIONS = 60


def reject_data(ticker, reason):
    print(f"ODRZUCONO {ticker}: {reason}")
    return None


def analyze_market(ticker):
    print("\n" + "=" * 45)
    print(f"ANALIZA: {ticker}")
    print("=" * 45)

    try:
        # 1. POBIERANIE DANYCH
        stock = yf.Ticker(ticker)

        data = stock.history(
            period="1y",
            interval="1d",
            auto_adjust=True
        )

        if data.empty:
            return reject_data(ticker, "Brak danych z Yahoo Finance.")

        if "Close" not in data.columns:
            return reject_data(ticker, "Brak kolumny Close.")

        # 2. CZYSZCZENIE DANYCH
        data = data.copy()
        data = data.sort_index()
        data = data[~data.index.duplicated(keep="last")]

        data["Close"] = pd.to_numeric(
            data["Close"],
            errors="coerce"
        )

        data = data.dropna(subset=["Close"])

        # Usuwanie nieprawidlowych cen
        data = data[
            data["Close"].map(
                lambda value: math.isfinite(float(value))
                and float(value) > 0
            )
        ]

        if "Volume" not in data.columns:
            data["Volume"] = 0

        data["Volume"] = pd.to_numeric(
            data["Volume"],
            errors="coerce"
        ).fillna(0)

        if (data["Volume"] < 0).any():
            return reject_data(ticker, "Ujemny wolumen.")

        if len(data) < MIN_SESSIONS:
            return reject_data(
                ticker,
                f"Za malo poprawnych sesji: {len(data)}."
            )

        # 3. KONTROLA AKTUALNOSCI
        latest = data.iloc[-1]
        latest_date = pd.Timestamp(latest.name).date()
        today = pd.Timestamp.now(tz="UTC").date()

        age_days = (today - latest_date).days

        if age_days < 0:
            return reject_data(
                ticker,
                "Data notowan znajduje sie w przyszlosci."
            )

        if age_days > MAX_DATA_AGE_DAYS:
            return reject_data(
                ticker,
                f"Nieaktualne notowania: {age_days} dni."
            )

        close = data["Close"]
        volume = data["Volume"]

        # 4. SREDNIE KROCZACE
        data["SMA20"] = close.rolling(20).mean()
        data["SMA50"] = close.rolling(50).mean()

        # 5. RSI
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

        # 6. MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        data["MACD"] = ema12 - ema26

        data["SIGNAL"] = data["MACD"].ewm(
            span=9,
            adjust=False
        ).mean()

        # 7. WOLUMEN
        data["AVG_VOLUME"] = volume.shift(1).rolling(20).mean()

        # 8. OSTATNIA SESJA
        latest = data.iloc[-1]
        previous = data.iloc[-2]

        price = float(latest["Close"])
        previous_price = float(previous["Close"])

        if previous_price <= 0:
            return reject_data(ticker, "Nieprawidlowa cena poprzedniej sesji.")

        sma20 = float(latest["SMA20"])
        sma50 = float(latest["SMA50"])
        rsi = float(latest["RSI"])
        macd = float(latest["MACD"])
        macd_signal = float(latest["SIGNAL"])

        required_values = [
            price, previous_price, sma20,
            sma50, rsi, macd, macd_signal
        ]

        if not all(math.isfinite(value) for value in required_values):
            return reject_data(
                ticker,
                "Nie mozna obliczyc wszystkich wskaznikow."
            )

        if not 0 <= rsi <= 100:
            return reject_data(ticker, "Nieprawidlowa wartosc RSI.")

        change = (price / previous_price - 1) * 100

        # 9. MOMENTUM
        momentum_5d = (
            price / float(close.iloc[-6]) - 1
        ) * 100

        momentum_20d = (
            price / float(close.iloc[-21]) - 1
        ) * 100

        # 10. ZMIENNOSC ROCZNA
        daily_returns = close.pct_change()

        volatility = (
            daily_returns.tail(20).std()
            * math.sqrt(252)
            * 100
        )

        volatility = float(volatility)

        if not math.isfinite(volatility):
            return reject_data(ticker, "Nieprawidlowa zmiennosc.")

        # 11. WOLUMEN
        avg_volume = float(latest["AVG_VOLUME"])
        current_volume = float(latest["Volume"])

        volume_ratio = None

        if math.isfinite(avg_volume) and avg_volume > 0:
            volume_ratio = current_volume / avg_volume

        # 12. TREND
        if price > sma20 > sma50:
            trend = "WZROSTOWY"
        elif price < sma20 < sma50:
            trend = "SPADKOWY"
        else:
            trend = "BOCZNY"

        # 13. POTWIERDZENIE SYGNALU
        confirmation = (
            (data["Close"] > data["SMA20"])
            & (data["SMA20"] > data["SMA50"])
            & (data["MACD"] > data["SIGNAL"])
        )

        confirmed_days = int(confirmation.tail(3).sum())

        # 14. SYSTEM PUNKTOWY
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

        # 15. SYGNAL
        if score >= 7 and confirmed_days >= 2:
            signal = "POZYTYWNY"
        elif score >= 4:
            signal = "NEUTRALNY"
        else:
            signal = "NEGATYWNY"

        # 16. RYZYKO
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

        # 17. OSTRZEZENIA
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

        if volume_ratio is not None and volume_ratio >= 1.5:
            warnings.append("WYSOKI WOLUMEN")

        if confirmed_days < 2:
            warnings.append("SYGNAL NIE MA POTWIERDZENIA")

        # 18. RAPORT
        print(f"Status danych: POPRAWNE")
        print(f"Data notowan: {latest_date}")
        print(f"Wiek danych: {age_days} dni")
        print(f"Liczba sesji: {len(data)}")

        print(f"\nCena: {price:.2f} PLN")
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
            "date": str(latest_date),
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
            "momentum_20d": round(momentum_20d, 2),
            "data_status": "OK",
            "data_age_days": age_days,
            "sessions_count": len(data),
            "warnings": warnings
        }

    except Exception as error:
        print(f"BLAD ANALIZY {ticker}: {error}")
        return None
