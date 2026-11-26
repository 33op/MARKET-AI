    import math
    import warnings
    from collections import Counter
    
    import pandas as pd
    import yfinance as yf
    
    
    VERSION = "16.3"
    
    INITIAL_CAPITAL = 20000.0
    MAX_POSITIONS = 3
    MAX_POSITION_WEIGHT = 0.30
    MIN_CASH_WEIGHT = 0.10

HISTORY_PERIOD = "5y"
MIN_SESSIONS = 100
MIN_CALENDAR_SESSIONS = 150

COMMISSION_RATE = 0.003
SLIPPAGE_RATE = 0.001

HOLDING_PERIODS = [1, 3, 5]
DEFAULT_HOLDING_SESSIONS = 3

BENCHMARK = "^WIG"


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def safe_float(value, default=0.0):
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def normalize_index(index):
    index = pd.to_datetime(index)

    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)

    return index.normalize()


def download_history(ticker, period=HISTORY_PERIOD):
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required = ["Close", "High", "Low", "Volume"]

        if not all(column in data.columns for column in required):
            return None

        data = data[required].copy()
        data.index = normalize_index(data.index)
        data = data[~data.index.duplicated(keep="last")]
        data = data.sort_index()

        for column in required:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

        data = data.dropna(subset=["Close"])

        if len(data) < MIN_SESSIONS:
            return None

        return data

    except Exception as error:
        print(f"BACKTEST {ticker}: błąd pobierania danych: {error}")
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
        (avg_gain == 0) & (avg_loss > 0),
        "RSI"
    ] = 0

    data["EMA12"] = close.ewm(
        span=12,
        adjust=False
    ).mean()

    data["EMA26"] = close.ewm(
        span=26,
        adjust=False
    ).mean()

    data["MACD"] = data["EMA12"] - data["EMA26"]

    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["AVG_VOLUME"] = volume.shift(1).rolling(20).mean()

    data["VOLUME_RATIO"] = (
        volume / data["AVG_VOLUME"].replace(0, float("nan"))
    )

    data["MOMENTUM_5"] = (
        close / close.shift(5) - 1
    ) * 100

    data["MOMENTUM_20"] = (
        close / close.shift(20) - 1
    ) * 100

    data["DAILY_CHANGE"] = close.pct_change() * 100

    data["VOLATILITY"] = (
        close.pct_change()
        .rolling(20)
        .std()
        * math.sqrt(252)
        * 100
    )

    data["SMA20_RISING"] = (
        data["SMA20"] > data["SMA20"].shift(5)
    )

    condition = (
        (close > data["SMA20"])
        & (data["SMA20"] > data["SMA50"])
        & (data["MACD"] > data["MACD_SIGNAL"])
    )

    data["CONFIRMED_DAYS"] = (
        condition.astype(int).rolling(3).sum()
    )

    return data


# ============================================================
# SYGNAŁ MARKET AI
# ============================================================

def calculate_signal(row):
    score = 0

    price = safe_float(row.get("Close"))
    sma20 = safe_float(row.get("SMA20"))
    sma50 = safe_float(row.get("SMA50"))

    rsi = safe_float(row.get("RSI"), 50)
    macd = safe_float(row.get("MACD"))
    macd_signal = safe_float(row.get("MACD_SIGNAL"))

    momentum5 = safe_float(row.get("MOMENTUM_5"))
    momentum20 = safe_float(row.get("MOMENTUM_20"))

    volume_ratio = safe_float(row.get("VOLUME_RATIO"))
    daily_change = safe_float(row.get("DAILY_CHANGE"))
    volatility = safe_float(row.get("VOLATILITY"))

    confirmed = int(
        safe_float(row.get("CONFIRMED_DAYS"))
    )

    if price > sma20:
        score += 1

    if sma20 > sma50:
        score += 1

    if bool(row.get("SMA20_RISING", False)):
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

    if volume_ratio >= 1.2 and daily_change > 0:
        score += 1

    if rsi > 80:
        score -= 2
    elif rsi > 70:
        score -= 1

    if daily_change >= 10:
        score -= 2
    elif daily_change >= 5:
        score -= 1

    if volume_ratio >= 4:
        score -= 1

    if volatility >= 60:
        score -= 2
    elif volatility >= 40:
        score -= 1

    score = max(0, min(10, score))

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

def apply_buy_cost(price, shares):
    gross = price * shares
    slippage = gross * SLIPPAGE_RATE
    commission = gross * COMMISSION_RATE

    return gross + slippage + commission


def apply_sell_cost(price, shares):
    gross = price * shares
    slippage = gross * SLIPPAGE_RATE
    commission = gross * COMMISSION_RATE

    return gross - slippage - commission


def calculate_buy_gross(price, shares):
    return price * shares


def calculate_sell_gross(price, shares):
    return price * shares


# ============================================================
# STATYSTYKI
# ============================================================

def max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    worst = 0.0

    for value in equity_curve:
        peak = max(peak, value)

        if peak > 0:
            drawdown = (value / peak - 1) * 100
            worst = min(worst, drawdown)

    return worst


def calculate_profit_factor(trades):
    profits = sum(
        trade["net_profit"]
        for trade in trades
        if trade["net_profit"] > 0
    )

    losses = abs(sum(
        trade["net_profit"]
        for trade in trades
        if trade["net_profit"] < 0
    ))

    if losses == 0:
        return float("inf") if profits > 0 else 0.0

    return profits / losses


def summarize_period(
    starting_capital,
    ending_capital,
    equity_curve,
    trades
):
    net_return = (
        (ending_capital / starting_capital) - 1
    ) * 100 if starting_capital else 0.0

    gross_profits = [
        trade["gross_profit"]
        for trade in trades
    ]

    net_profits = [
        trade["net_profit"]
        for trade in trades
    ]

    winners = [
        trade for trade in trades
        if trade["net_profit"] > 0
    ]

    gross_return = (
        sum(gross_profits) / starting_capital
    ) * 100 if starting_capital else 0.0

    average_trade = (
        sum(net_profits) / len(net_profits)
        if net_profits else 0.0
    )

    win_rate = (
        len(winners) / len(trades) * 100
        if trades else 0.0
    )

    return {
        "starting_capital": starting_capital,
        "ending_capital": ending_capital,
        "capital": ending_capital,
        "return": net_return,
        "gross_return": gross_return,
        "drawdown": max_drawdown(equity_curve),
        "trades": len(trades),
        "win_rate": win_rate,
        "average_trade": average_trade,
        "profit_factor": calculate_profit_factor(trades),
        "equity_curve": equity_curve,
        "gross_profit": sum(gross_profits),
        "net_profit": sum(net_profits),
        "trades_by_ticker": dict(
            Counter(trade["ticker"] for trade in trades)
        )
    }


# ============================================================
# PRZYGOTOWANIE RYNKU
# ============================================================

def prepare_market(instruments):
    market = {}

    print("\nBACKTEST: pobieranie danych historycznych...")

    for index, ticker in enumerate(instruments, start=1):
        print(
            f"[{index}/{len(instruments)}] "
            f"Pobieranie {ticker}..."
        )

        history = download_history(ticker)

        if history is None:
            continue

        try:
            history = calculate_indicators(history)
            market[ticker] = history
        except Exception as error:
            print(f"BACKTEST {ticker}: błąd wskaźników: {error}")

    return market


def build_market_calendar(market):
    dates = set()

    for data in market.values():
        dates.update(data.index)

    return sorted(dates)


# ============================================================
# WARTOŚĆ PORTFELA
# ============================================================

def get_position_value(position):
    price = safe_float(position.get("last_price"))

    if price <= 0:
        return 0.0

    return apply_sell_cost(
        price,
        position["shares"]
    )


def calculate_portfolio_value(cash, positions):
    return cash + sum(
        get_position_value(position)
        for position in positions.values()
    )


# ============================================================
# SYMULACJA PORTFELA
# ============================================================

def run_portfolio_simulation(
    market,
    calendar,
    start_date,
    end_date,
    holding_sessions
):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    equity_curve = []

    last_prices = {}
    last_dates = {}

    simulation_dates = [
        date for date in calendar
        if start_date <= date <= end_date
    ]

    for date in simulation_dates:

        # ----------------------------------------------------
        # Aktualizacja ostatnich dostępnych cen
        # ----------------------------------------------------

        for ticker, data in market.items():
            available = data.loc[data.index <= date]

            if available.empty:
                continue

            latest = available.iloc[-1]

            last_prices[ticker] = safe_float(
                latest["Close"]
            )

            last_dates[ticker] = available.index[-1]

        # ----------------------------------------------------
        # Aktualizacja cen posiadanych pozycji
        # ----------------------------------------------------

        for ticker, position in list(positions.items()):
            if ticker in last_prices:
                position["last_price"] = last_prices[ticker]

        # ----------------------------------------------------
        # Sprzedaż pozycji po określonym czasie
        # ----------------------------------------------------

        for ticker, position in list(positions.items()):

            data = market[ticker]

            ticker_dates = data.index[
                data.index > position["entry_date"]
            ]

            if len(ticker_dates) < holding_sessions:
                continue

            exit_date = ticker_dates[
                holding_sessions - 1
            ]

            if date < exit_date:
                continue

            exit_price = safe_float(
                data.loc[exit_date, "Close"]
            )

            shares = position["shares"]

            gross_buy = calculate_buy_gross(
                position["entry_price"],
                shares
            )

            gross_sell = calculate_sell_gross(
                exit_price,
                shares
            )

            buy_cost = apply_buy_cost(
                position["entry_price"],
                shares
            ) - gross_buy

            sell_cost = gross_sell - apply_sell_cost(
                exit_price,
                shares
            )

            net_buy = gross_buy + buy_cost
            net_sell = gross_sell - sell_cost

            gross_profit = gross_sell - gross_buy
            net_profit = net_sell - net_buy

            cash += net_sell

            trades.append({
                "ticker": ticker,
                "entry_date": position["entry_date"],
                "exit_date": exit_date,
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "shares": shares,
                "gross_profit": gross_profit,
                "net_profit": net_profit,
                "buy_cost": buy_cost,
                "sell_cost": sell_cost
            })

            del positions[ticker]

        # ----------------------------------------------------
        # Szukanie nowych sygnałów
        # ----------------------------------------------------

        candidates = []

        for ticker, data in market.items():

            if ticker in positions:
                continue

            if ticker not in last_dates:
                continue

            available = data.loc[data.index < date]

            # Potrzebujemy dwóch wcześniejszych sesji:
            # aktualnej oceny i poprzedniej oceny.
            if len(available) < 2:
                continue

            current_row = available.iloc[-1]
            previous_row = available.iloc[-2]

            current_signal = calculate_signal(current_row)
            previous_signal = calculate_signal(previous_row)

            # Nowy sygnał: wcześniej nie był pozytywny,
            # a teraz jest pozytywny i potwierdzony.
            is_new_signal = (
                current_signal["signal"] == "POZYTYWNY"
                and previous_signal["signal"] != "POZYTYWNY"
            )

            if not is_new_signal:
                continue

            entry_price = safe_float(
                market[ticker].loc[date, "Close"]
                if date in market[ticker].index
                else 0
            )

            if entry_price <= 0:
                continue

            candidates.append({
                "ticker": ticker,
                "score": current_signal["score"],
                "confirmed": current_signal["confirmed"],
                "entry_price": entry_price
            })

        candidates.sort(
            key=lambda item: (
                item["score"],
                item["confirmed"]
            ),
            reverse=True
        )

        # ----------------------------------------------------
        # Otwieranie nowych pozycji
        # ----------------------------------------------------

        for candidate in candidates:

            if len(positions) >= MAX_POSITIONS:
                break

            ticker = candidate["ticker"]
            price = candidate["entry_price"]

            if ticker in positions:
                continue

            portfolio_value = calculate_portfolio_value(
                cash,
                positions
            )

            max_position_value = (
                portfolio_value * MAX_POSITION_WEIGHT
            )

            minimum_cash = (
                portfolio_value * MIN_CASH_WEIGHT
            )

            available_cash = cash - minimum_cash

            budget = min(
                max_position_value,
                available_cash
            )

            if budget <= 0:
                continue

            # Uwzględniamy prowizję i poślizg już przy zakupie.
            estimated_unit_cost = (
                price
                * (1 + COMMISSION_RATE + SLIPPAGE_RATE)
            )

            shares = math.floor(
                budget / estimated_unit_cost
            )

            if shares <= 0:
                continue

            total_cost = apply_buy_cost(
                price,
                shares
            )

            if total_cost > available_cash:
                continue

            cash -= total_cost

            positions[ticker] = {
                "ticker": ticker,
                "entry_date": date,
                "entry_price": price,
                "shares": shares,
                "last_price": price,
                "holding_sessions": holding_sessions
            }

        # ----------------------------------------------------
        # Wycena portfela na koniec sesji
        # ----------------------------------------------------

        equity = calculate_portfolio_value(
            cash,
            positions
        )

        equity_curve.append(equity)

    # ========================================================
    # ZAMKNIĘCIE POZOSTAŁYCH POZYCJI NA KONIEC OKRESU
    # ========================================================

    for ticker, position in list(positions.items()):

        data = market[ticker]

        available = data.loc[data.index <= end_date]

        if available.empty:
            continue

        exit_date = available.index[-1]
        exit_price = safe_float(
            available.iloc[-1]["Close"]
        )

        shares = position["shares"]

        gross_buy = calculate_buy_gross(
            position["entry_price"],
            shares
        )

        gross_sell = calculate_sell_gross(
            exit_price,
            shares
        )

        buy_cost = apply_buy_cost(
            position["entry_price"],
            shares
        ) - gross_buy

        sell_cost = gross_sell - apply_sell_cost(
            exit_price,
            shares
        )

        net_buy = gross_buy + buy_cost
        net_sell = gross_sell - sell_cost

        gross_profit = gross_sell - gross_buy
        net_profit = net_sell - net_buy

        cash += net_sell

        trades.append({
            "ticker": ticker,
            "entry_date": position["entry_date"],
            "exit_date": exit_date,
            "entry_price": position["entry_price"],
            "exit_price": exit_price,
            "shares": shares,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "buy_cost": buy_cost,
            "sell_cost": sell_cost
        })

    ending_capital = cash

    return summarize_period(
        INITIAL_CAPITAL,
        ending_capital,
        equity_curve,
        trades
    )


# ============================================================
# BENCHMARK WIG
# ============================================================

def benchmark_return(start_date, end_date):
    try:
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")

        # Yahoo Finance traktuje end jako datę wyłączną.
        end = (
            pd.Timestamp(end_date) + pd.Timedelta(days=2)
        ).strftime("%Y-%m-%d")

        data = yf.download(
            BENCHMARK,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        if "Close" not in data.columns:
            return None

        close = data["Close"].dropna()

        if len(close) < 2:
            return None

        return (
            (safe_float(close.iloc[-1]) /
             safe_float(close.iloc[0])) - 1
        ) * 100

    except Exception as error:
        print(f"BACKTEST: błąd benchmarku WIG: {error}")
        return None


# ============================================================
# GŁÓWNY BACKTEST
# ============================================================

def run_backtest(instruments):
    print("\n" + "=" * 65)
    print(f"MARKET AI v{VERSION} | BACKTEST PORTFELOWY")
    print("=" * 65)

    market = prepare_market(instruments)

    if not market:
        print("BACKTEST: brak danych do symulacji.")

        return {
            "version": VERSION,
            "instruments": 0,
            "calendar_sessions": 0,
            "train": {},
            "out_of_sample": {},
            "benchmark": None,
            "benchmark_name": "WIG",
            "test_start": None,
            "test_end": None,
            "costs_included": True,
            "holding_comparison": {}
        }

    calendar = build_market_calendar(market)

    if len(calendar) < MIN_CALENDAR_SESSIONS:
        print(
            "BACKTEST: za mało wspólnych sesji "
            "do wiarygodnego podziału."
        )

        return {
            "version": VERSION,
            "instruments": len(market),
            "calendar_sessions": len(calendar),
            "train": {},
            "out_of_sample": {},
            "benchmark": None,
            "benchmark_name": "WIG",
            "test_start": None,
            "test_end": None,
            "costs_included": True,
            "holding_comparison": {}
        }

    # Pierwsze 60% kalendarza pomijamy jako okres
    # rozgrzewkowy wskaźników i treningowy.
    # Pozostałe 40% to test poza próbą.

    split_index = int(len(calendar) * 0.70)

    train_start = calendar[60]
    train_end = calendar[split_index - 1]

    test_start = calendar[split_index]
    test_end = calendar[-1]

    print(f"\nInstrumenty: {len(market)}")
    print(f"Liczba sesji: {len(calendar)}")
    print(
        f"Okres treningowy: {train_start.date()} - "
        f"{train_end.date()}"
    )
    print(
        f"Okres testowy: {test_start.date()} - "
        f"{test_end.date()}"
    )

    holding_comparison = {}

    for holding in HOLDING_PERIODS:

        print(
            f"\nBACKTEST: testowanie trzymania "
            f"pozycji przez {holding} sesji..."
        )

        train_result = run_portfolio_simulation(
            market,
            calendar,
            train_start,
            train_end,
            holding
        )

        test_result = run_portfolio_simulation(
            market,
            calendar,
            test_start,
            test_end,
            holding
        )

        holding_comparison[holding] = {
            "train": train_result,
            "out_of_sample": test_result
        }

    # Domyślny wariant raportu to 3 sesje.
    # Pozostałe warianty są pokazane osobno,
    # aby można było je porównać.

    selected = holding_comparison[
        DEFAULT_HOLDING_SESSIONS
    ]

    benchmark = benchmark_return(
        test_start,
        test_end
    )

    report = {
        "version": VERSION,
        "instruments": len(market),
        "calendar_sessions": len(calendar),
        "train": selected["train"],
        "out_of_sample": selected["out_of_sample"],
        "benchmark": benchmark,
        "benchmark_name": "WIG",
        "test_start": test_start,
        "test_end": test_end,
        "costs_included": True,
        "commission_rate": COMMISSION_RATE,
        "slippage_rate": SLIPPAGE_RATE,
        "holding_sessions": DEFAULT_HOLDING_SESSIONS,
        "holding_comparison": holding_comparison
    }

    # ========================================================
    # RAPORT KONSOLI
    # ========================================================

    print("\n" + "=" * 65)
    print(f"WYNIKI BACKTESTU MARKET AI v{VERSION}")
    print("=" * 65)

    for holding in HOLDING_PERIODS:

        result = holding_comparison[holding]["out_of_sample"]

        print(
            f"\nTRZYMANIE: {holding} SESJI"
        )

        print(
            f"Kapitał końcowy: "
            f"{result['ending_capital']:,.2f} PLN"
        )

        print(
            f"Wynik netto: "
            f"{result['return']:+.2f}%"
        )

        print(
            f"Wynik brutto: "
            f"{result['gross_return']:+.2f}%"
        )

        print(
            f"Maksymalne obsunięcie: "
            f"{result['drawdown']:.2f}%"
        )

        print(
            f"Liczba transakcji: "
            f"{result['trades']}"
        )

        print(
            f"Skuteczność: "
            f"{result['win_rate']:.2f}%"
        )

        print(
            f"Średni wynik transakcji: "
            f"{result['average_trade']:+.2f} PLN"
        )

        pf = result["profit_factor"]

        if math.isinf(pf):
            pf_text = "brak stratnych transakcji"
        else:
            pf_text = f"{pf:.2f}"

        print(f"Profit factor: {pf_text}")

    print("\n" + "=" * 65)
    print("PORÓWNANIE Z WIG")
    print("=" * 65)

    if benchmark is None:
        print("Brak danych benchmarku WIG.")
    else:
        print(f"WIG: {benchmark:+.2f}%")

    print(
        f"MARKET AI ({DEFAULT_HOLDING_SESSIONS} sesje): "
        f"{selected['out_of_sample']['return']:+.2f}%"
    )

    print("\nKoszty uwzględnione:")
    print(f"Prowizja: {COMMISSION_RATE * 100:.2f}%")
    print(f"Poślizg: {SLIPPAGE_RATE * 100:.2f}%")

    print("\nBACKTEST ZAKOŃCZONY.")
    print("=" * 65)

    return report
