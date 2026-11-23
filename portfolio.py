
import json
import os
import math
from collections import defaultdict, Counter

import yfinance as yf
import pandas as pd


VERSION = "13.0"

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "portfolio_history.json")

BENCHMARK = "^WIG"
HISTORY_PERIOD = "2y"
HORIZONS = [1, 5, 10]


def valid_number(value):
    try:
        number = float(value)
        return math.isfinite(number)
    except (TypeError, ValueError):
        return False


def load_history():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            history = json.load(file)

        return history if isinstance(history, list) else []

    except (OSError, json.JSONDecodeError):
        print("PORTFEL: Nie mozna odczytac historii.")
        return []


def save_history(history):
    os.makedirs(DATA_DIR, exist_ok=True)

    temporary_file = DATA_FILE + ".tmp"

    with open(temporary_file, "w", encoding="utf-8") as file:
        json.dump(
            history,
            file,
            ensure_ascii=False,
            indent=4,
            allow_nan=False
        )

    os.replace(temporary_file, DATA_FILE)


def parse_date(value):
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


def create_snapshot(history, allocations, summary, results):
    if not results:
        return history, "Brak analiz do zapisania."

    result_by_ticker = {
        item.get("ticker"): item
        for item in results
        if item.get("ticker")
    }

    if allocations:
        relevant_dates = []

        for allocation in allocations:
            result = result_by_ticker.get(
                allocation["ticker"]
            )

            if result is None:
                return history, (
                    "Nie mozna zapisac portfela: "
                    "brak danych wybranego instrumentu."
                )

            date = parse_date(result.get("date"))

            if date is None:
                return history, (
                    "Nie mozna ustalic daty analizy."
                )

            relevant_dates.append(date)

        if len(set(relevant_dates)) != 1:
            return history, (
                "Pominieto zapis: wybrane instrumenty "
                "maja rozne daty notowan."
            )

        snapshot_date = relevant_dates[0]

    else:
        available_dates = [
            parse_date(item.get("date"))
            for item in results
        ]

        available_dates = [
            date for date in available_dates
            if date is not None
        ]

        if not available_dates:
            return history, "Brak poprawnej daty analizy."

        snapshot_date = Counter(
            available_dates
        ).most_common(1)[0][0]

    date_text = snapshot_date.isoformat()

    if any(
        item.get("date") == date_text
        for item in history
        if isinstance(item, dict)
    ):
        return history, (
            f"Portfel z dnia {date_text} juz istnieje."
        )

    positions = []

    for allocation in allocations:
        price = allocation.get("price")
        shares = allocation.get("shares")

        if (
            not valid_number(price)
            or not valid_number(shares)
            or float(price) <= 0
            or float(shares) < 0
        ):
            continue

        positions.append({
            "ticker": allocation["ticker"],
            "price": float(price),
            "shares": int(shares)
        })

    cash = float(summary.get("remaining", 20000.0))

    snapshot = {
        "date": date_text,
        "capital": 20000.0,
        "cash": cash,
        "positions": positions,
        "forward_returns": {}
    }

    history.append(snapshot)

    return history, (
        f"Zapisano portfel z dnia {date_text}."
    )


def download_prices(tickers):
    tickers = sorted(set(tickers))

    if not tickers:
        return {}

    print(
        f"\nPORTFEL: Pobieranie danych dla "
        f"{len(tickers)} instrumentow."
    )

    try:
        data = yf.download(
            tickers=tickers,
            period=HISTORY_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column"
        )

    except Exception as error:
        print(f"PORTFEL: Blad pobierania danych: {error}")
        return {}

    if data is None or data.empty:
        print("PORTFEL: Brak danych historycznych.")
        return {}

    output = {}

    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ("Close", ticker) in data.columns:
                    close = data[("Close", ticker)]

                elif (ticker, "Close") in data.columns:
                    close = data[(ticker, "Close")]

                elif "Close" in data.columns.get_level_values(0):
                    close = data["Close"][ticker]

                elif "Close" in data.columns.get_level_values(1):
                    close = data.xs(
                        "Close",
                        level=1,
                        axis=1
                    )[ticker]

                else:
                    continue

            else:
                if "Close" not in data.columns:
                    continue

                close = data["Close"]

            close = pd.to_numeric(
                close,
                errors="coerce"
            ).dropna()

            close = close[close > 0]

            index = pd.to_datetime(close.index)

            if index.tz is not None:
                index = index.tz_localize(None)

            close.index = index.date

            close = close[
                ~close.index.duplicated(keep="last")
            ]

            if len(close) >= 2:
                output[ticker] = {
                    date: float(price)
                    for date, price in close.items()
                    if valid_number(price) and float(price) > 0
                }

        except Exception as error:
            print(
                f"PORTFEL: Nie mozna odczytac "
                f"{ticker}: {error}"
            )

    return output


def evaluate_snapshot(snapshot, price_data):
    snapshot_date = parse_date(snapshot.get("date"))

    if snapshot_date is None:
        return

    positions = snapshot.get("positions", [])

    tickers = [
        position["ticker"]
        for position in positions
    ]

    required_tickers = tickers + [BENCHMARK]

    if any(
        ticker not in price_data
        for ticker in required_tickers
    ):
        return

    for ticker in required_tickers:
        if snapshot_date not in price_data[ticker]:
            return

    common_dates = None

    for ticker in required_tickers:
        dates = {
            date
            for date in price_data[ticker]
            if date > snapshot_date
        }

        common_dates = (
            dates
            if common_dates is None
            else common_dates.intersection(dates)
        )

    if not common_dates:
        return

    future_dates = sorted(common_dates)

    if not isinstance(snapshot.get("forward_returns"), dict):
        snapshot["forward_returns"] = {}

    capital = float(snapshot.get("capital", 20000.0))
    cash = float(snapshot.get("cash", capital))

    for horizon in HORIZONS:
        key = f"{horizon}_sessions"

        if key in snapshot["forward_returns"]:
            continue

        if len(future_dates) < horizon:
            continue

        target_date = future_dates[horizon - 1]

        portfolio_value = cash

        for position in positions:
            ticker = position["ticker"]
            shares = float(position["shares"])

            future_price = price_data[ticker].get(
                target_date
            )

            if future_price is None:
                portfolio_value = None
                break

            portfolio_value += shares * future_price

        if portfolio_value is None or capital <= 0:
            continue

        benchmark_start = price_data[BENCHMARK][snapshot_date]
        benchmark_end = price_data[BENCHMARK][target_date]

        portfolio_return = (
            portfolio_value / capital - 1
        ) * 100

        benchmark_return = (
            benchmark_end / benchmark_start - 1
        ) * 100

        excess_return = (
            portfolio_return - benchmark_return
        )

        snapshot["forward_returns"][key] = {
            "portfolio": round(portfolio_return, 2),
            "benchmark": round(benchmark_return, 2),
            "excess": round(excess_return, 2),
            "target_date": target_date.isoformat()
        }

        print(
            f"PORTFEL {snapshot['date']} | "
            f"{horizon} sesji | "
            f"Portfel: {portfolio_return:+.2f}% | "
            f"WIG: {benchmark_return:+.2f}% | "
            f"Przewaga: {excess_return:+.2f}%"
        )


def evaluate_history(history):
    pending = []

    for snapshot in history:
        if not isinstance(snapshot, dict):
            continue

        returns = snapshot.get("forward_returns", {})

        if not isinstance(returns, dict):
            snapshot["forward_returns"] = {}
            returns = snapshot["forward_returns"]

        if len(returns) < len(HORIZONS):
            pending.append(snapshot)

    if not pending:
        return history

    tickers = {BENCHMARK}

    for snapshot in pending:
        for position in snapshot.get("positions", []):
            ticker = position.get("ticker")

            if ticker:
                tickers.add(ticker)

    price_data = download_prices(tickers)

    if not price_data:
        return history

    for snapshot in pending:
        evaluate_snapshot(snapshot, price_data)

    return history


def get_statistics(history):
    statistics = {}

    for horizon in HORIZONS:
        key = f"{horizon}_sessions"

        portfolio_values = []
        benchmark_values = []
        excess_values = []

        for snapshot in history:
            returns = snapshot.get("forward_returns", {})

            if not isinstance(returns, dict):
                continue

            result = returns.get(key)

            if not isinstance(result, dict):
                continue

            portfolio = result.get("portfolio")
            benchmark = result.get("benchmark")
            excess = result.get("excess")

            if not all(
                valid_number(value)
                for value in (portfolio, benchmark, excess)
            ):
                continue

            portfolio_values.append(float(portfolio))
            benchmark_values.append(float(benchmark))
            excess_values.append(float(excess))

        def summarize(values):
            if not values:
                return {
                    "count": 0,
                    "average": None,
                    "median": None,
                    "positive": None
                }

            return {
                "count": len(values),
                "average": round(
                    sum(values) / len(values),
                    2
                ),
                "median": round(
                    float(pd.Series(values).median()),
                    2
                ),
                "positive": round(
                    sum(value > 0 for value in values)
                    / len(values) * 100,
                    1
                )
            }

        statistics[key] = {
            "portfolio": summarize(portfolio_values),
            "benchmark": summarize(benchmark_values),
            "excess": summarize(excess_values)
        }

    return statistics


def format_statistics(statistics):
    lines = []

    for horizon, groups in statistics.items():
        lines.append(f"**{horizon}**")

        portfolio = groups["portfolio"]
        benchmark = groups["benchmark"]
        excess = groups["excess"]

        if portfolio["count"] == 0:
            lines.append("Za malo danych.")
            continue

        lines.append(
            f"Portfel: {portfolio['average']:+.2f}% "
            f"(mediana {portfolio['median']:+.2f}%)"
        )

        lines.append(
            f"WIG: {benchmark['average']:+.2f}% "
            f"(mediana {benchmark['median']:+.2f}%)"
        )

        lines.append(
            f"Przewaga nad WIG: {excess['average']:+.2f}% "
            f"({excess['count']} prob)"
        )

    return "\n".join(lines)


def update_portfolio_history(allocations, summary, results):
    history = load_history()

    history, status = create_snapshot(
        history,
        allocations,
        summary,
        results
    )

    history = evaluate_history(history)

    save_history(history)

    statistics = get_statistics(history)

    print("\n" + "=" * 65)
    print("MARKET AI 13.0 - PORTFEL VS WIG")
    print("=" * 65)
    print(status)
    print(format_statistics(statistics))
    print("=" * 65)

    return statistics, status
