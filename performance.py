
import json
import os
import math
from collections import defaultdict

import yfinance as yf
import pandas as pd


VERSION = "13.0"
EVALUATION_VERSION = 2

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "performance.json")

HORIZONS = [1, 5, 10]

SIGNALS = [
    "POZYTYWNY",
    "NEUTRALNY",
    "NEGATYWNY"
]

HISTORY_PERIOD = "2y"


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

        if not isinstance(history, list):
            print("Historia ma nieprawidlowy format.")
            return []

        return history

    except (json.JSONDecodeError, OSError):
        print("Nie mozna odczytac historii analiz.")
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


def add_snapshots(history, results):
    existing = {
        (item.get("ticker"), item.get("date"))
        for item in history
        if isinstance(item, dict)
    }

    added = 0

    for result in results:
        ticker = result.get("ticker")
        date = result.get("date")
        price = result.get("price")

        if not ticker or not date or not valid_number(price):
            continue

        price = float(price)

        if price <= 0:
            continue

        key = (ticker, date)

        if key in existing:
            continue

        signal = result.get("signal", "NEUTRALNY")

        if signal not in SIGNALS:
            signal = "NEUTRALNY"

        history.append({
            "ticker": ticker,
            "date": date,
            "price": price,
            "score": result.get("score", 0),
            "signal": signal,
            "risk": result.get("risk", "NIEZNANE"),
            "confirmed_days": result.get("confirmed_days", 0),
            "forward_returns": {},
            "evaluation_version": EVALUATION_VERSION
        })

        existing.add(key)
        added += 1

    print(f"\nZapisano nowych analiz: {added}")

    return history


def prepare_market_data(ticker):
    data = yf.Ticker(ticker).history(
        period=HISTORY_PERIOD,
        interval="1d",
        auto_adjust=True
    )

    if data.empty or "Close" not in data.columns:
        return None

    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data["Close"] = pd.to_numeric(
        data["Close"],
        errors="coerce"
    )

    data = data.dropna(subset=["Close"])
    data = data[data["Close"] > 0]

    if data.empty:
        return None

    data["SessionDate"] = data.index.date

    return data


def evaluate_ticker_records(ticker, records):
    print(f"\nUZUPELNIANIE HISTORII: {ticker}")

    try:
        data = prepare_market_data(ticker)

        if data is None:
            print(f"Brak danych historycznych: {ticker}")
            return

        dates = list(data["SessionDate"])
        prices = list(data["Close"])

        price_by_date = dict(zip(dates, prices))

        for item in records:
            try:
                snapshot_date = pd.Timestamp(
                    item["date"]
                ).date()

                if snapshot_date not in price_by_date:
                    print(
                        f"{ticker}: brak dokladnej sesji "
                        f"dla {snapshot_date}"
                    )
                    continue

                stored_price = float(item["price"])

                if not valid_number(stored_price) or stored_price <= 0:
                    continue

                actual_price = float(
                    price_by_date[snapshot_date]
                )

                difference = abs(
                    actual_price / stored_price - 1
                ) * 100

                if difference > 1.0:
                    print(
                        f"{ticker}: OSTRZEZENIE - cena zapisana "
                        f"rozni sie od danych Yahoo o "
                        f"{difference:.2f}%"
                    )
                    continue

                start_position = dates.index(snapshot_date)

                future_prices = prices[
                    start_position + 1:
                ]

                for horizon in HORIZONS:
                    key = f"{horizon}_sessions"

                    if key in item["forward_returns"]:
                        continue

                    if len(future_prices) < horizon:
                        continue

                    future_price = float(
                        future_prices[horizon - 1]
                    )

                    change = (
                        future_price / actual_price - 1
                    ) * 100

                    if not math.isfinite(change):
                        continue

                    item["forward_returns"][key] = round(
                        change,
                        2
                    )

                    print(
                        f"{ticker}: {horizon} sesji = "
                        f"{change:+.2f}%"
                    )

            except (ValueError, TypeError, KeyError, IndexError) as error:
                print(
                    f"Blad pojedynczego wpisu {ticker}: {error}"
                )

    except Exception as error:
        print(f"Blad historii {ticker}: {error}")


def evaluate_history(history):
    grouped = defaultdict(list)

    for item in history:
        if not isinstance(item, dict):
            continue

        if not all(
            key in item
            for key in ("ticker", "date", "price")
        ):
            continue

        if not valid_number(item.get("price")):
            continue

        if float(item["price"]) <= 0:
            continue

        if not isinstance(item.get("forward_returns"), dict):
            item["forward_returns"] = {}

        if item.get("evaluation_version", 0) < EVALUATION_VERSION:
            item["forward_returns"] = {}
            item["evaluation_version"] = EVALUATION_VERSION

        if len(item["forward_returns"]) < len(HORIZONS):
            grouped[item["ticker"]].append(item)

    for ticker, records in grouped.items():
        evaluate_ticker_records(ticker, records)

    return history


def get_statistics(history):
    statistics = {}

    for signal in SIGNALS:
        statistics[signal] = {}

        for horizon in HORIZONS:
            key = f"{horizon}_sessions"

            values = []

            for item in history:
                if not isinstance(item, dict):
                    continue

                if item.get("signal") != signal:
                    continue

                returns = item.get("forward_returns", {})

                if not isinstance(returns, dict):
                    continue

                if key not in returns:
                    continue

                value = returns[key]

                if not valid_number(value):
                    continue

                values.append(float(value))

            if not values:
                statistics[signal][key] = {
                    "count": 0,
                    "average": None,
                    "median": None,
                    "positive": None,
                    "negative": None
                }
                continue

            statistics[signal][key] = {
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
                ),

                "negative": round(
                    sum(value < 0 for value in values)
                    / len(values) * 100,
                    1
                )
            }

    return statistics


def print_statistics(statistics):
    print("\n" + "=" * 65)
    print(f"MARKET AI {VERSION} - SKUTECZNOSC SYGNALOW")
    print("=" * 65)

    for signal, horizons in statistics.items():
        print(f"\nSYGNAL: {signal}")

        for horizon, values in horizons.items():
            if values["count"] == 0:
                print(f"{horizon}: za malo danych")
                continue

            print(
                f"{horizon}: "
                f"srednia: {values['average']:+.2f}% | "
                f"mediana: {values['median']:+.2f}% | "
                f"dodatnie: {values['positive']}% | "
                f"ujemne: {values['negative']}% | "
                f"prob: {values['count']}"
            )

    print("\n" + "=" * 65)
    print("UWAGA:")
    print(
        "Wyniki historyczne nie gwarantuja przyszlych rezultatow."
    )
    print(
        "Powtarzajace sie analizy tej samej spolki "
        "nie sa niezaleznymi probami."
    )
    print("=" * 65)


def update_performance(results):
    history = load_history()

    history = add_snapshots(history, results)

    history = evaluate_history(history)

    save_history(history)

    statistics = get_statistics(history)

    print_statistics(statistics)

    return statistics
