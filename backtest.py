
import math
import warnings

import pandas as pd
import yfinance as yf


VERSION = "16.0"

INITIAL_CAPITAL = 20000.0
MAX_POSITIONS = 3
MAX_POSITION_WEIGHT = 0.30
MIN_CASH_WEIGHT = 0.10

HISTORY_PERIOD = "5y"
MIN_SESSIONS = 100

COMMISSION_RATE = 0.003
SLIPPAGE_RATE = 0.001

# Pozycja jest utrzymywana do kolejnego zamknięcia sesji.
# Decyzja z dnia T jest wykonywana na zamknięciu T+1.
HOLDING_SESSIONS = 1

BENCHMARK = "^WIG"


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

        if data is None or data.empty or "Close" not in data:
            return None

        data = data.copy().sort_index()
        data = data[~data.index.duplicated(keep="last")]

        data["Close"] = pd.to_numeric(
            data["Close"], errors="coerce"
        )
        data = data.dropna(subset=["Close"])
        data = data[data["Close"] > 0]

        if "Volume" not in data:
            data["Volume"] = 0

        data["Volume"] = pd.to_numeric(
            data["Volume"], errors="coerce"
        ).fillna(0)

        if len(data) < MIN_SESSIONS:
            return None

        return data

    except Exception as error:
        print(f"BACKTEST {ticker}: {error}")
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
        (avg_loss == 0) & (avg_gain > 0), "RSI"
    ] = 100

    data.loc[
        (avg_loss == 0) & (avg_gain == 0), "RSI"
    ] = 50

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    data["MACD"] = ema12 - ema26
    data["SIGNAL"] = data["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    data["AVG_VOLUME"] = volume.shift(1).rolling(20).mean()
    data["MOMENTUM_5"] = close.pct_change(5) * 100
    data["MOMENTUM_20"] = close.pct_change(20) * 100
    data["DAILY_CHANGE"] = close.pct_change() * 100

    data["VOLATILITY"] = (
        close.pct_change().rolling(20).std()
        * math.sqrt(252) * 100
    )

    data["SMA20_RISING"] = (
        data["SMA20"] > data["SMA20"].shift(5)
    )

    confirmation = (
        (close > data["SMA20"])
        & (data["SMA20"] > data["SMA50"])
        & (data["MACD"] > data["SIGNAL"])
    )

    data["CONFIRMED_DAYS"] = confirmation.rolling(3).sum()

    return data


def calculate_signal(row):
    required = [
        "Close", "SMA20", "SMA50", "RSI",
        "MACD", "SIGNAL", "MOMENTUM_5",
        "MOMENTUM_20", "DAILY_CHANGE",
        "VOLATILITY", "CONFIRMED_DAYS"
    ]

    if any(pd.isna(row.get(key)) for key in required):
        return None

    price = safe_float(row["Close"])
    sma20 = safe_float(row["SMA20"])
    sma50 = safe_float(row["SMA50"])
    rsi = safe_float(row["RSI"])
    macd = safe_float(row["MACD"])
    macd_signal = safe_float(row["SIGNAL"])
    momentum5 = safe_float(row["MOMENTUM_5"])
    momentum20 = safe_float(row["MOMENTUM_20"])
    change = safe_float(row["DAILY_CHANGE"])
    volatility = safe_float(row["VOLATILITY"])
    confirmed = safe_float(row["CONFIRMED_DAYS"])

    values = [
        price, sma20, sma50, rsi, macd,
        macd_signal, momentum5, momentum20,
        change, volatility, confirmed
    ]

    if any(value is None for value in values):
        return None

    if price <= 0 or sma20 <= 0 or sma50 <= 0:
        return None

    if not 0 <= rsi <= 100:
        return None

    volume = safe_float(row["Volume"])
    average_volume = safe_float(row["AVG_VOLUME"])

    volume_ratio = None

    if (
        volume is not None
        and average_volume is not None
        and average_volume > 0
    ):
        volume_ratio = volume / average_volume

    score = 0

    if price > sma20:
        score += 1

    if sma20 > sma50:
        score += 1

    if bool(row["SMA20_RISING"]):
        score += 1

    if macd > macd_signal:
        score += 2

    if 50 <= rsi <= 65:
        score += 2
    elif 40 <= rsi < 50 or 65 < rsi <= 70:
        score += 1

    if momentum5 > 0:
        score += 1

    if momentum20 > 0:
        score += 1

    if volume_ratio is not None and volume_ratio >= 1.2 and change > 0:
        score += 1

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

    score = max(0, min(10, score - penalty))
    confirmed = int(confirmed)

    if score >= 7 and confirmed >= 2:
        signal = "POZYTYWNY"
    elif score >= 4:
        signal = "NEUTRALNY"
    else:
        signal = "NEGATYWNY"

    return {
        "score": score,
        "signal": signal,
        "confirmed": confirmed
    }


def apply_buy_cost(price):
    return price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)


def apply_sell_cost(price):
    return price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)


def max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0

    series = pd.Series(equity_curve, dtype=float)
    peaks = series.cummax()
    drawdowns = (series / peaks - 1) * 100

    return float(drawdowns.min())


def prepare_market(instruments):
    market = {}

    for ticker in instruments:
        print(f"Pobieranie danych: {ticker}")

        data = download_history(ticker)

        if data is None:
            continue

        market[ticker] = calculate_indicators(data)

    return market


def run_portfolio_simulation(market, start_index, end_index):
    cash = INITIAL_CAPITAL
    positions = {}
    equity_curve = []
    trade_returns = []
    completed_trades = 0

    dates = sorted(set().union(*[
        set(data.index) for data in market.values()
    ]))

    dates = dates[start_index:end_index]

    if len(dates) < 3:
        return None

    for date in dates:
        # Najpierw wycena aktualnych pozycji.
        current_prices = {}

        for ticker, position in list(positions.items()):
            data = market[ticker]

            if date in data.index:
                price = safe_float(data.loc[date, "Close"])
                if price is not None and price > 0:
                    current_prices[ticker] = price

        # Sprzedaż pozycji, które osiągnęły termin wyjścia.
        for ticker, position in list(positions.items()):
            if date >= position["exit_date"]:
                price = current_prices.get(ticker)

                if price is None:
                    continue

                proceeds = (
                    position["shares"] * apply_sell_cost(price)
                )

                cash += proceeds

                entry_value = position["entry_value"]
                trade_return = (
                    proceeds / entry_value - 1
                ) * 100

                trade_returns.append(trade_return)
                completed_trades += 1
                del positions[ticker]

        # Wartość portfela przed nowymi zakupami.
        position_value = 0.0

        for ticker, position in positions.items():
            price = current_prices.get(ticker)

            if price is None:
                price = position["last_price"]
            else:
                position["last_price"] = price

            position_value += position["shares"] * price

        equity = cash + position_value

        # Sygnały z poprzedniej sesji. Nie używamy przyszłych danych.
        candidates = []

        for ticker, data in market.items():
            if ticker in positions:
                continue

            previous_rows = data.loc[data.index < date]

            if previous_rows.empty:
                continue

            row = previous_rows.iloc[-1]
            signal = calculate_signal(row)

            if signal and signal["signal"] == "POZYTYWNY":
                candidates.append((
                    ticker,
                    signal["score"],
                    signal["confirmed"]
                ))

        candidates.sort(
            key=lambda item: (item[1], item[2]),
            reverse=True
        )

        # Limit 3 pozycji i maksymalnie 30% wartości portfela.
        for ticker, score, confirmed in candidates:
            if len(positions) >= MAX_POSITIONS:
                break

            if ticker in positions:
                continue

            data = market[ticker]

            if date not in data.index:
                continue

            price = safe_float(data.loc[date, "Close"])

            if price is None or price <= 0:
                continue

            equity = cash + sum(
                p["shares"] * p["last_price"]
                for p in positions.values()
            )

            max_allocation = equity * MAX_POSITION_WEIGHT
            cash_floor = equity * MIN_CASH_WEIGHT
            available_cash = max(0.0, cash - cash_floor)

            allocation = min(max_allocation, available_cash)

            if allocation <= 0:
                break

            buy_price = apply_buy_cost(price)
            shares = allocation / buy_price

            if shares <= 0:
                continue

            actual_cost = shares * buy_price

            if actual_cost > available_cash:
                continue

            cash -= actual_cost

            future_dates = [
                d for d in data.index if d > date
            ]

            if len(future_dates) < HOLDING_SESSIONS:
                cash += actual_cost
                continue

            exit_date = future_dates[HOLDING_SESSIONS - 1]

            positions[ticker] = {
                "shares": shares,
                "entry_value": actual_cost,
                "entry_date": date,
                "exit_date": exit_date,
                "last_price": price
            }

        # Wycena na koniec sesji.
        position_value = 0.0

        for ticker, position in positions.items():
            data = market[ticker]

            if date in data.index:
                price = safe_float(data.loc[date, "Close"])
                if price is not None:
                    position["last_price"] = price

            position_value += (
                position["shares"] * position["last_price"]
            )

        equity_curve.append(cash + position_value)

    # Na końcu testu wyceniamy otwarte pozycje.
    final_equity = cash

    for position in positions.values():
        final_equity += (
            position["shares"] * position["last_price"]
        )

    if equity_curve:
        equity_curve[-1] = final_equity

    total_return = (
        final_equity / INITIAL_CAPITAL - 1
    ) * 100

    return {
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": final_equity,
        "return": total_return,
        "max_drawdown": max_drawdown(equity_curve),
        "trades": completed_trades,
        "win_rate": (
            sum(value > 0 for value in trade_returns)
            / len(trade_returns) * 100
            if trade_returns else None
        ),
        "average_trade": (
            sum(trade_returns) / len(trade_returns)
            if trade_returns else None
        ),
        "equity_curve": equity_curve
    }


def benchmark_return(start_date, end_date):
    try:
        data = yf.Ticker(BENCHMARK).history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True
        )

        if data is None or data.empty or "Close" not in data:
            return None

        close = data["Close"].dropna()

        if len(close) < 2 or close.iloc[0] <= 0:
            return None

        return (
            close.iloc[-1] / close.iloc[0] - 1
        ) * 100

    except Exception as error:
        print(f"BENCHMARK: {error}")
        return None


def run_backtest(instruments):
    warnings.filterwarnings("ignore")

    print("=" * 65)
    print(f"MARKET AI v{VERSION} - PORTFOLIO BACKTEST")
    print("=" * 65)

    print(f"Kapitał: {INITIAL_CAPITAL:.2f} PLN")
    print(f"Maksymalna liczba pozycji: {MAX_POSITIONS}")
    print(f"Limit jednej pozycji: {MAX_POSITION_WEIGHT * 100:.0f}%")
    print(f"Minimalna gotówka: {MIN_CASH_WEIGHT * 100:.0f}%")
    print(f"Prowizja na stronę: {COMMISSION_RATE * 100:.2f}%")
    print(f"Poślizg na stronę: {SLIPPAGE_RATE * 100:.2f}%")

    market = prepare_market(instruments)

    if not market:
        print("Brak danych do backtestu.")
        return {"version": VERSION, "error": "Brak danych"}

    common_dates = sorted(set.intersection(*[
        set(data.index) for data in market.values()
    ]))

    if len(common_dates) < 150:
        print("Za mało wspólnych sesji.")
        return {"version": VERSION, "error": "Za mało danych"}

    split = int(len(common_dates) * 0.70)

    # Okres projektowy i okres poza próbą.
    train = run_portfolio_simulation(
        market, 60, split
    )

    test = run_portfolio_simulation(
        market, split, len(common_dates) - 1
    )

    start_date = common_dates[split]
    end_date = common_dates[-1]

    benchmark = benchmark_return(start_date, end_date)

    result = {
        "version": VERSION,
        "instruments": len(market),
        "train": train,
        "out_of_sample": test,
        "benchmark": benchmark,
        "benchmark_name": BENCHMARK,
        "test_start": str(start_date.date()),
        "test_end": str(end_date.date()),
        "costs_included": True
    }

    print("\nWYNIKI OKRESU PROJEKTOWEGO")

    if train:
        print(f"Wynik: {train['return']:+.2f}%")
        print(f"Obsunięcie: {train['max_drawdown']:.2f}%")
        print(f"Transakcje: {train['trades']}")

    print("\nWYNIKI POZA PRÓBĄ")

    if test:
        print(f"Kapitał końcowy: {test['final_equity']:.2f} PLN")
        print(f"Wynik: {test['return']:+.2f}%")
        print(f"Maksymalne obsunięcie: {test['max_drawdown']:.2f}%")
        print(f"Transakcje: {test['trades']}")

        if test["win_rate"] is not None:
            print(f"Skuteczność transakcji: {test['win_rate']:.1f}%")

        if test["average_trade"] is not None:
            print(f"Średni wynik transakcji: {test['average_trade']:+.2f}%")

    print("\nPORÓWNANIE Z WIG")

    if benchmark is not None and test:
        print(f"MARKET AI: {test['return']:+.2f}%")
        print(f"WIG: {benchmark:+.2f}%")
        print(f"Różnica: {test['return'] - benchmark:+.2f} pp")
    else:
        print("Brak danych benchmarku.")

    print("\nUWAGI:")
    print("- Wyniki poza próbą dotyczą późniejszego okresu danych.")
    print("- Sygnał z poprzedniej sesji jest wykonywany na kolejnym zamknięciu.")
    print("- Koszty transakcyjne są założeniami symulacji.")
    print("- Brak gwarancji realizacji po cenie zamknięcia.")
    print("- To nie jest pełna symulacja mechaniki SIGG.")

    print("=" * 65)

    return result
