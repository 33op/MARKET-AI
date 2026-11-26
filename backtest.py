import math
import warnings
from collections import Counter

import pandas as pd
import yfinance as yf


VERSION = "16.4"

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
    """Pobiera skorygowane dane OHLCV potrzebne do symulacji."""
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

        required = ["Open", "Close", "High", "Low", "Volume"]

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

        data = data.dropna(subset=["Open", "Close"])

        data = data[
            (data["Open"] > 0)
            & (data["Close"] > 0)
        ]

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

def transaction_cost_breakdown(price, shares):
    gross = max(0.0, safe_float(price) * int(shares))
    return {
        "gross": gross,
        "slippage": gross * SLIPPAGE_RATE,
        "commission": gross * COMMISSION_RATE
    }


def apply_buy_cost(price, shares):
    costs = transaction_cost_breakdown(price, shares)
    return costs["gross"] + costs["slippage"] + costs["commission"]


def apply_sell_cost(price, shares):
    costs = transaction_cost_breakdown(price, shares)
    return costs["gross"] - costs["slippage"] - costs["commission"]


def calculate_buy_gross(price, shares):
    return safe_float(price) * int(shares)


def calculate_sell_gross(price, shares):
    return safe_float(price) * int(shares)


# ============================================================
# STATYSTYKI
# ============================================================

def max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0

    peak = safe_float(equity_curve[0])
    worst = 0.0

    for raw_value in equity_curve:
        value = safe_float(raw_value)

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

    gross_profit = sum(
        trade["gross_profit"] for trade in trades
    )

    net_profit = sum(
        trade["net_profit"] for trade in trades
    )

    total_commission = sum(
        trade["buy_commission"] + trade["sell_commission"]
        for trade in trades
    )

    total_slippage = sum(
        trade["buy_slippage"] + trade["sell_slippage"]
        for trade in trades
    )

    winners = [
        trade for trade in trades
        if trade["net_profit"] > 0
    ]

    average_trade = (
        net_profit / len(trades)
        if trades else 0.0
    )

    win_rate = (
        len(winners) / len(trades) * 100
        if trades else 0.0
    )

    gross_return = (
        gross_profit / starting_capital
    ) * 100 if starting_capital else 0.0

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
        "gross_profit": gross_profit,
        "net_profit": net_profit,
        "total_commission": total_commission,
        "total_slippage": total_slippage,
        "total_transaction_costs": total_commission + total_slippage,
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
            market[ticker] = calculate_indicators(history)
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

    # Wycena likwidacyjna: zakładamy sprzedaż po ostatnim
    # dostępnym zamknięciu, z uwzględnieniem kosztów sprzedaży.
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
# ROZLICZANIE TRANSAKCJI
# ============================================================

def close_position(
    ticker,
    position,
    exit_date,
    exit_price,
    cash,
    trades
):
    shares = position["shares"]

    buy_costs = transaction_cost_breakdown(
        position["entry_price"],
        shares
    )

    sell_costs = transaction_cost_breakdown(
        exit_price,
        shares
    )

    gross_buy = buy_costs["gross"]
    gross_sell = sell_costs["gross"]

    net_buy = (
        gross_buy
        + buy_costs["commission"]
        + buy_costs["slippage"]
    )

    net_sell = (
        gross_sell
        - sell_costs["commission"]
        - sell_costs["slippage"]
    )

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
        "buy_cost": (
            buy_costs["commission"] + buy_costs["slippage"]
        ),
        "sell_cost": (
            sell_costs["commission"] + sell_costs["slippage"]
        ),
        "buy_commission": buy_costs["commission"],
        "sell_commission": sell_costs["commission"],
        "buy_slippage": buy_costs["slippage"],
        "sell_slippage": sell_costs["slippage"]
    })

    return cash


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
    if holding_sessions < 1:
        raise ValueError("holding_sessions musi wynosić co najmniej 1.")

    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    equity_curve = [INITIAL_CAPITAL]

    # Ostatnie znane zamknięcia. Aktualizujemy je dopiero
    # po wykonaniu transakcji danego dnia, aby nie używać
    # dzisiejszego Close przy decyzji podejmowanej na Open.
    last_prices = {}

    simulation_dates = [
        date for date in calendar
        if start_date <= date <= end_date
    ]

    for date in simulation_dates:

        # ----------------------------------------------------
        # SYGNAŁY: wyłącznie dane dostępne przed dzisiejszą sesją
        # ----------------------------------------------------

        candidates = []

        for ticker, data in market.items():

            if ticker in positions:
                continue

            # Wejście możliwe tylko wtedy, gdy instrument
            # rzeczywiście ma notowanie w tej sesji.
            if date not in data.index:
                continue

            entry_price = safe_float(data.at[date, "Open"])

            if entry_price <= 0:
                continue

            available = data.loc[data.index < date]

            if len(available) < 2:
                continue

            current_row = available.iloc[-1]
            previous_row = available.iloc[-2]

            current_signal = calculate_signal(current_row)
            previous_signal = calculate_signal(previous_row)

            is_new_signal = (
                current_signal["signal"] == "POZYTYWNY"
                and previous_signal["signal"] != "POZYTYWNY"
            )

            if not is_new_signal:
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
        # ZAKUPY: wykonanie na Open
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

            estimated_unit_cost = (
                price * (
                    1 + COMMISSION_RATE + SLIPPAGE_RATE
                )
            )

            shares = math.floor(
                budget / estimated_unit_cost
            )

            if shares <= 0:
                continue

            total_cost = apply_buy_cost(price, shares)

            if total_cost > available_cash:
                continue

            cash -= total_cost

            ticker_dates = market[ticker].index
            future_sessions = ticker_dates[
                ticker_dates >= date
            ]

            planned_exit_date = (
                future_sessions[holding_sessions - 1]
                if len(future_sessions) >= holding_sessions
                else None
            )

            positions[ticker] = {
                "ticker": ticker,
                "entry_date": date,
                "entry_price": price,
                "shares": shares,
                "last_price": price,
                "planned_exit_date": planned_exit_date,
                "score": candidate["score"],
                "confirmed": candidate["confirmed"]
            }

        # ----------------------------------------------------
        # SPRZEDAŻE: wykonanie na Close
        # Holding 1 oznacza Open -> Close tej samej sesji.
        # Holding 3 oznacza Open pierwszej sesji -> Close
        # trzeciej sesji instrumentu.
        # ----------------------------------------------------

        for ticker, position in list(positions.items()):

            exit_date = position.get("planned_exit_date")

            if exit_date is None or date < exit_date:
                continue

            data = market[ticker]

            if date not in data.index:
                continue

            exit_price = safe_float(data.at[date, "Close"])

            if exit_price <= 0:
                continue

            cash = close_position(
                ticker,
                position,
                date,
                exit_price,
                cash,
                trades
            )

            del positions[ticker]

        # ----------------------------------------------------
        # WYCENA NA KONIEC SESJI
        # ----------------------------------------------------

        for ticker, data in market.items():
            if date in data.index:
                close_price = safe_float(data.at[date, "Close"])

                if close_price > 0:
                    last_prices[ticker] = close_price

        for ticker, position in positions.items():
            if ticker in last_prices:
                position["last_price"] = last_prices[ticker]

        equity = calculate_portfolio_value(cash, positions)
        equity_curve.append(equity)

    # ========================================================
    # KOŃCOWA LIKWIDACJA POZYCJI NA KONIEC OKRESU
    # ========================================================

    for ticker, position in list(positions.items()):

        data = market[ticker]
        available = data.loc[data.index <= end_date]

        if available.empty:
            # Nie powinno się zdarzyć przy poprawnym kalendarzu,
            # ale nie wyceniamy nieistniejącej ceny jako zera.
            continue

        exit_date = available.index[-1]
        exit_price = safe_float(available.iloc[-1]["Close"])

        if exit_price <= 0:
            continue

        cash = close_position(
            ticker,
            position,
            exit_date,
            exit_price,
            cash,
            trades
        )

        del positions[ticker]

    ending_capital = cash

    # Końcowy punkt krzywej uwzględnia koszty likwidacji.
    equity_curve.append(ending_capital)

    result = summarize_period(
        INITIAL_CAPITAL,
        ending_capital,
        equity_curve,
        trades
    )

    # Kontrola księgowa: przy zamkniętych wszystkich pozycjach
    # zmiana kapitału musi odpowiadać sumie wyników netto.
    reconciliation = (
        ending_capital
        - INITIAL_CAPITAL
        - result["net_profit"]
    )

    result["reconciliation_error"] = reconciliation

    return result


# ============================================================
# BENCHMARK WIG
# ============================================================

def benchmark_return(start_date, end_date):
    try:
        start = pd.Timestamp(start_date).strftime("%Y-%m-%d")

        # Yahoo Finance traktuje parametr end jako datę wyłączną.
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

        close = pd.to_numeric(
            data["Close"],
            errors="coerce"
        ).dropna()

        if len(close) < 2:
            return None

        first = safe_float(close.iloc[0])
        last = safe_float(close.iloc[-1])

        if first <= 0:
            return None

        return ((last / first) - 1) * 100

    except Exception as error:
        print(f"BACKTEST: błąd benchmarku WIG: {error}")
        return None


# ============================================================
# GŁÓWNY BACKTEST
# ============================================================

def empty_backtest_report(instruments_count=0, calendar_sessions=0):
    return {
        "version": VERSION,
        "instruments": instruments_count,
        "calendar_sessions": calendar_sessions,
        "train": {},
        "out_of_sample": {},
        "benchmark": None,
        "benchmark_name": "WIG",
        "test_start": None,
        "test_end": None,
        "costs_included": True,
        "commission_rate": COMMISSION_RATE,
        "slippage_rate": SLIPPAGE_RATE,
        "holding_sessions": DEFAULT_HOLDING_SESSIONS,
        "holding_comparison": {}
    }


def run_backtest(instruments):
    print("\n" + "=" * 65)
    print(f"MARKET AI v{VERSION} | BACKTEST PORTFELOWY")
    print("=" * 65)

    market = prepare_market(instruments)

    if not market:
        print("BACKTEST: brak danych do symulacji.")
        return empty_backtest_report()

    calendar = build_market_calendar(market)

    if len(calendar) < MIN_CALENDAR_SESSIONS:
        print(
            "BACKTEST: za mało sesji do wiarygodnego podziału."
        )
        return empty_backtest_report(
            len(market),
            len(calendar)
        )

    # Pierwsze 60 sesji służy jako rozgrzewka wskaźników.
    # Od sesji 61 do granicy 70% kalendarza trwa trening.
    # Ostatnie 30% kalendarza stanowi test poza próbą.
    split_index = int(len(calendar) * 0.70)

    if split_index <= 60 or split_index >= len(calendar):
        print("BACKTEST: nieprawidłowy podział danych.")
        return empty_backtest_report(
            len(market),
            len(calendar)
        )

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
        f"Okres testowy OOS: {test_start.date()} - "
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

        print(f"\nTRZYMANIE: {holding} SESJI")
        print(
            f"Kapitał końcowy: "
            f"{result['ending_capital']:,.2f} PLN"
        )
        print(f"Wynik netto: {result['return']:+.2f}%")
        print(f"Wynik brutto: {result['gross_return']:+.2f}%")
        print(
            f"Maksymalne obsunięcie: "
            f"{result['drawdown']:.2f}%"
        )
        print(f"Liczba transakcji: {result['trades']}")
        print(f"Skuteczność: {result['win_rate']:.2f}%")
        print(
            f"Średni wynik transakcji: "
            f"{result['average_trade']:+.2f} PLN"
        )
        print(
            f"Prowizje łącznie: "
            f"{result['total_commission']:,.2f} PLN"
        )
        print(
            f"Poślizg łącznie: "
            f"{result['total_slippage']:,.2f} PLN"
        )
        print(
            f"Wszystkie koszty: "
            f"{result['total_transaction_costs']:,.2f} PLN"
        )
        print(
            f"Kontrola księgowa: "
            f"{result['reconciliation_error']:.8f} PLN"
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
