
import math
import warnings

import pandas as pd
import yfinance as yf


VERSION = "16.1"

INITIAL_CAPITAL = 20000.0
MAX_POSITIONS = 3
MAX_POSITION_WEIGHT = 0.30
MIN_CASH_WEIGHT = 0.10

HISTORY_PERIOD = "5y"
MIN_SESSIONS = 100
MIN_CALENDAR_SESSIONS = 150

COMMISSION_RATE = 0.003
SLIPPAGE_RATE = 0.001

HOLDING_SESSIONS = 1

BENCHMARK = "^WIG"


# ============================================================
# PODSTAWOWE FUNKCJE
# ============================================================

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
            data["Close"],
            errors="coerce"
        )

        data = data.dropna(subset=["Close"])
        data = data[data["Close"] > 0]

        if "Volume" not in data:
            data["Volume"] = 0

        data["Volume"] = pd.to_numeric(
            data["Volume"],
            errors="coerce"
        ).fillna(0)

        if len(data) < MIN_SESSIONS:
            return None

        return data

    except Exception as error:
        print(f"BACKTEST {ticker}: {error}")
        return None


# ============================================================
# WSKAŹNIKI
# ============================================================

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

    data["AVG_VOLUME"] = (
        volume.shift(1).rolling(20).mean()
    )

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

    data["CONFIRMED_DAYS"] = (
        confirmation.rolling(3).sum()
    )

    return data


# ============================================================
# GENEROWANIE SYGNAŁÓW
# ============================================================

def calculate_signal(row):
    required = [
        "Close",
        "SMA20",
        "SMA50",
        "RSI",
        "MACD",
        "SIGNAL",
        "MOMENTUM_5",
        "MOMENTUM_20",
        "DAILY_CHANGE",
        "VOLATILITY",
        "CONFIRMED_DAYS"
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
        price,
        sma20,
        sma50,
        rsi,
        macd,
        macd_signal,
        momentum5,
        momentum20,
        change,
        volatility,
        confirmed
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

    if (
        volume_ratio is not None
        and volume_ratio >= 1.2
        and change > 0
    ):
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


# ============================================================
# KOSZTY TRANSAKCJI
# ============================================================

def apply_buy_cost(price):
    return price * (1 + SLIPPAGE_RATE) * (1 + COMMISSION_RATE)


def apply_sell_cost(price):
    return price * (1 - SLIPPAGE_RATE) * (1 - COMMISSION_RATE)


# ============================================================
# STATYSTYKI
# ============================================================

def max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0

    series = pd.Series(
        equity_curve,
        dtype=float
    )

    peaks = series.cummax()

    drawdowns = (
        series / peaks - 1
    ) * 100

    return float(drawdowns.min())


def calculate_profit_factor(trade_returns):
    profits = sum(
        value for value in trade_returns
        if value > 0
    )

    losses = abs(sum(
        value for value in trade_returns
        if value < 0
    ))

    if losses == 0:
        return float("inf") if profits > 0 else None

    return profits / losses


# ============================================================
# POBIERANIE RYNKU
# ============================================================

def prepare_market(instruments):
    market = {}

    rejected = 0

    for ticker in instruments:
        print(f"Pobieranie danych: {ticker}")

        data = download_history(ticker)

        if data is None:
            rejected += 1
            print(f"Odrzucono {ticker}: niewystarczające dane.")
            continue

        market[ticker] = calculate_indicators(data)

    print("\nPODSUMOWANIE POBIERANIA")
    print(f"Poprawne instrumenty: {len(market)}")
    print(f"Odrzucone instrumenty: {rejected}")

    return market


def build_market_calendar(market):
    """
    Tworzy kalendarz będący sumą dat wszystkich instrumentów.

    Nie wymaga, aby każda spółka miała notowania w każdej sesji.
    """

    all_dates = set()

    for data in market.values():
        all_dates.update(data.index)

    return sorted(all_dates)


# ============================================================
# WYCENA PORTFELA
# ============================================================

def get_position_value(positions, market, date):
    value = 0.0

    for ticker, position in positions.items():
        data = market[ticker]

        if date in data.index:
            price = safe_float(
                data.loc[date, "Close"]
            )

            if price is not None and price > 0:
                position["last_price"] = price

        price = position["last_price"]

        # Wartość likwidacyjna uwzględnia koszt sprzedaży.
        value += (
            position["shares"]
            * apply_sell_cost(price)
        )

    return value


# ============================================================
# SYMULACJA PORTFELA
# ============================================================

def run_portfolio_simulation(market, dates):
    cash = INITIAL_CAPITAL

    positions = {}

    equity_curve = [INITIAL_CAPITAL]

    trade_returns = []

    completed_trades = 0

    if len(dates) < 3:
        return None

    simulation_end = dates[-1]

    for date in dates:

        # ----------------------------------------------------
        # 1. AKTUALIZACJA CEN
        # ----------------------------------------------------

        current_prices = {}

        for ticker, position in positions.items():
            data = market[ticker]

            if date in data.index:
                price = safe_float(
                    data.loc[date, "Close"]
                )

                if price is not None and price > 0:
                    current_prices[ticker] = price
                    position["last_price"] = price

        # ----------------------------------------------------
        # 2. SPRZEDAŻ PO TERMINIE WYJŚCIA
        # ----------------------------------------------------

        for ticker, position in list(positions.items()):

            if date < position["exit_date"]:
                continue

            price = current_prices.get(ticker)

            if price is None:
                continue

            proceeds = (
                position["shares"]
                * apply_sell_cost(price)
            )

            cash += proceeds

            trade_return = (
                proceeds / position["entry_value"] - 1
            ) * 100

            trade_returns.append(trade_return)

            completed_trades += 1

            del positions[ticker]

        # ----------------------------------------------------
        # 3. WARTOŚĆ PORTFELA
        # ----------------------------------------------------

        position_value = get_position_value(
            positions,
            market,
            date
        )

        equity = cash + position_value

        # ----------------------------------------------------
        # 4. SYGNAŁY Z POPRZEDNIEJ SESJI
        # ----------------------------------------------------

        candidates = []

        for ticker, data in market.items():

            if ticker in positions:
                continue

            # Szukamy wyłącznie danych sprzed aktualnej sesji.
            previous_rows = data.loc[
                data.index < date
            ]

            if previous_rows.empty:
                continue

            row = previous_rows.iloc[-1]

            signal = calculate_signal(row)

            if (
                signal is not None
                and signal["signal"] == "POZYTYWNY"
            ):
                candidates.append((
                    ticker,
                    signal["score"],
                    signal["confirmed"]
                ))

        candidates.sort(
            key=lambda item: (
                item[1],
                item[2]
            ),
            reverse=True
        )

        # ----------------------------------------------------
        # 5. OTWIERANIE POZYCJI
        # ----------------------------------------------------

        for ticker, score, confirmed in candidates:

            if len(positions) >= MAX_POSITIONS:
                break

            if ticker in positions:
                continue

            data = market[ticker]

            if date not in data.index:
                continue

            price = safe_float(
                data.loc[date, "Close"]
            )

            if price is None or price <= 0:
                continue

            # Nie otwieramy pozycji, jeśli nie będzie
            # możliwe jej zamknięcie przed końcem testu.
            future_dates = [
                future_date
                for future_date in data.index
                if date < future_date <= simulation_end
            ]

            if len(future_dates) < HOLDING_SESSIONS:
                continue

            exit_date = future_dates[
                HOLDING_SESSIONS - 1
            ]

            current_position_value = get_position_value(
                positions,
                market,
                date
            )

            equity = cash + current_position_value

            max_allocation = (
                equity * MAX_POSITION_WEIGHT
            )

            cash_floor = (
                equity * MIN_CASH_WEIGHT
            )

            available_cash = max(
                0.0,
                cash - cash_floor
            )

            allocation = min(
                max_allocation,
                available_cash
            )

            if allocation <= 0:
                break

            buy_price = apply_buy_cost(price)

            # Symulacja całkowitej liczby jednostek.
            shares = math.floor(
                allocation / buy_price
            )

            if shares <= 0:
                continue

            actual_cost = shares * buy_price

            if actual_cost > available_cash:
                continue

            cash -= actual_cost

            positions[ticker] = {
                "shares": shares,
                "entry_value": actual_cost,
                "entry_date": date,
                "exit_date": exit_date,
                "last_price": price,
                "entry_price": buy_price,
                "signal_score": score,
                "confirmed": confirmed
            }

        # ----------------------------------------------------
        # 6. DZIENNA WYCENA
        # ----------------------------------------------------

        position_value = get_position_value(
            positions,
            market,
            date
        )

        equity_curve.append(
            cash + position_value
        )

    # ========================================================
    # 7. ZAMKNIĘCIE POZYCJI NA KOŃCU OKRESU
    # ========================================================

    for ticker, position in list(positions.items()):

        data = market[ticker]

        available_prices = data.loc[
            data.index <= simulation_end,
            "Close"
        ].dropna()

        if available_prices.empty:
            continue

        last_price = safe_float(
            available_prices.iloc[-1]
        )

        if last_price is None or last_price <= 0:
            last_price = position["last_price"]

        proceeds = (
            position["shares"]
            * apply_sell_cost(last_price)
        )

        cash += proceeds

        trade_return = (
            proceeds / position["entry_value"] - 1
        ) * 100

        trade_returns.append(trade_return)

        completed_trades += 1

    positions.clear()

    final_equity = cash

    if equity_curve:
        equity_curve[-1] = final_equity

    total_return = (
        final_equity / INITIAL_CAPITAL - 1
    ) * 100

    winning_trades = sum(
        value > 0 for value in trade_returns
    )

    losing_trades = sum(
        value < 0 for value in trade_returns
    )

    return {
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": final_equity,
        "return": total_return,
        "max_drawdown": max_drawdown(equity_curve),
        "trades": completed_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": (
            winning_trades / len(trade_returns) * 100
            if trade_returns else None
        ),
        "average_trade": (
            sum(trade_returns) / len(trade_returns)
            if trade_returns else None
        ),
        "profit_factor": calculate_profit_factor(
            trade_returns
        ),
        "equity_curve": equity_curve
    }


# ============================================================
# BENCHMARK WIG
# ============================================================

def benchmark_return(start_date, end_date):
    try:
        data = yf.Ticker(BENCHMARK).history(
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True
        )

        if (
            data is None
            or data.empty
            or "Close" not in data
        ):
            return None

        close = data["Close"].dropna()

        if len(close) < 2:
            return None

        first_price = safe_float(close.iloc[0])
        last_price = safe_float(close.iloc[-1])

        if (
            first_price is None
            or last_price is None
            or first_price <= 0
        ):
            return None

        return (
            last_price / first_price - 1
        ) * 100

    except Exception as error:
        print(f"BENCHMARK: {error}")
        return None


# ============================================================
# GŁÓWNY BACKTEST
# ============================================================

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
    print(f"Historia: {HISTORY_PERIOD}")

    market = prepare_market(instruments)

    if not market:
        print("Brak danych do backtestu.")

        return {
            "version": VERSION,
            "error": "Brak danych"
        }

    # Kalendarz to suma dat, a nie przecięcie.
    calendar = build_market_calendar(market)

    print(f"\nLiczba sesji w kalendarzu: {len(calendar)}")

    if len(calendar) < MIN_CALENDAR_SESSIONS:
        print("Za mało sesji w kalendarzu.")

        return {
            "version": VERSION,
            "error": "Za mało danych"
        }

    # Chronologiczny podział 70/30.
    split = int(len(calendar) * 0.70)

    if split <= 60 or len(calendar) - split < 30:
        print("Nie można poprawnie podzielić danych.")

        return {
            "version": VERSION,
            "error": "Nieprawidłowy podział danych"
        }

    train_dates = calendar[60:split]
    test_dates = calendar[split:]

    if len(train_dates) < 30 or len(test_dates) < 30:
        print("Za krótkie okresy symulacji.")

        return {
            "version": VERSION,
            "error": "Za krótkie okresy symulacji"
        }

    print("\nOKRES PROJEKTOWY")
    print(f"Sesje: {len(train_dates)}")
    print(f"Od: {train_dates[0]}")
    print(f"Do: {train_dates[-1]}")

    print("\nOKRES POZA PRÓBĄ")
    print(f"Sesje: {len(test_dates)}")
    print(f"Od: {test_dates[0]}")
    print(f"Do: {test_dates[-1]}")

    # Każdy okres rozpoczyna z kapitałem początkowym.
    train = run_portfolio_simulation(
        market,
        train_dates
    )

    test = run_portfolio_simulation(
        market,
        test_dates
    )

    benchmark = benchmark_return(
        test_dates[0],
        test_dates[-1]
    )

    result = {
        "version": VERSION,
        "instruments": len(market),
        "calendar_sessions": len(calendar),
        "train": train,
        "out_of_sample": test,
        "benchmark": benchmark,
        "benchmark_name": BENCHMARK,
        "test_start": str(test_dates[0].date()),
        "test_end": str(test_dates[-1].date()),
        "costs_included": True
    }

    # ========================================================
    # RAPORT
    # ========================================================

    print("\n" + "=" * 65)
    print("WYNIKI OKRESU PROJEKTOWEGO")
    print("=" * 65)

    if train:
        print(f"Wynik: {train['return']:+.2f}%")
        print(f"Kapitał końcowy: {train['final_equity']:.2f} PLN")
        print(f"Obsunięcie: {train['max_drawdown']:.2f}%")
        print(f"Transakcje: {train['trades']}")

    print("\n" + "=" * 65)
    print("WYNIKI POZA PRÓBĄ")
    print("=" * 65)

    if test:
        print(f"Kapitał końcowy: {test['final_equity']:.2f} PLN")
        print(f"Wynik: {test['return']:+.2f}%")
        print(f"Maksymalne obsunięcie: {test['max_drawdown']:.2f}%")
        print(f"Transakcje: {test['trades']}")

        if test["win_rate"] is not None:
            print(f"Skuteczność: {test['win_rate']:.1f}%")

        if test["average_trade"] is not None:
            print(f"Średni wynik transakcji: {test['average_trade']:+.2f}%")

        if test["profit_factor"] is not None:
            print(f"Profit factor: {test['profit_factor']:.2f}")

    print("\n" + "=" * 65)
    print("PORÓWNANIE Z WIG")
    print("=" * 65)

    if benchmark is not None and test:
        print(f"MARKET AI: {test['return']:+.2f}%")
        print(f"WIG: {benchmark:+.2f}%")
        print(
            f"Różnica: "
            f"{test['return'] - benchmark:+.2f} pp"
        )
    else:
        print("Brak danych benchmarku.")

    print("\nUWAGI:")
    print("- Kalendarz obejmuje sesje dostępne dla analizowanych instrumentów.")
    print("- Sygnał jest obliczany na podstawie poprzedniej sesji.")
    print("- Wejście następuje na zamknięciu aktualnej sesji.")
    print("- Pozycje są zamykane po określonej liczbie sesji.")
    print("- Otwarte pozycje są zamykane na końcu badanego okresu.")
    print("- Prowizja i poślizg są założeniami symulacji.")
    print("- Wyniki nie odwzorowują dokładnie mechaniki SIGG.")
    print("- Wyniki historyczne nie gwarantują przyszłych rezultatów.")

    print("=" * 65)

    return result


if __name__ == "__main__":
    from instruments import INSTRUMENTS

    run_backtest(INSTRUMENTS)
