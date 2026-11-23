
import json
import os
from collections import defaultdict

import yfinance as yf
import pandas as pd

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "performance.json")

HORIZONS = [1, 5, 10]
SIGNALS = ["POZYTYWNY", "NEUTRALNY", "NEGATYWNY"]


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
        json.dump(history, file, ensure_ascii=False, indent=4)

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

        if not ticker or not date or price is None:
            continue

        key = (ticker, date)

        if key in existing:
            continue

        try:
            price = float(price)

            if price <= 0:
                continue

        except (TypeError, ValueError):
            continue

        history.append({
            "ticker": ticker,
            "date": date,
            "price": price,
            "score": result.get("score", 0),
            "signal": result.get("signal", "NEUTRALNY"),
            "risk": result.get("risk", "NIEZNANE"),
            "confirmed_days": result.get("confirmed_days", 0),
            "forward_returns": {}
        })

        existing.add(key)
        added += 1

    print(f"\nZapisano nowych analiz: {added}")

    return history


def evaluate_history(history):
    grouped = defaultdict(list)

    for item in history:
        if not isinstance(item, dict):
            continue

        if not all(key in item for key in ("ticker", "date", "price")):
            continue

        if not isinstance(item.get("forward_returns"), dict):
            item["forward_returns"] = {}

        if len(item["forward_returns"]) < len(HORIZONS):
            grouped[item["ticker"]].append(item)

    for ticker, records in grouped.items():
        try:
            print(f"\nUZUPELNIANIE HISTORII: {ticker}")

            data = yf.Ticker(ticker).history(
                period="1y",
                interval="1d",
                auto_adjust=True
            )

            if data.empty or "Close" not in data.columns:
                print(f"Brak danych historycznych: {ticker}")
                continue

            data = data.sort_index()
            data = data[~data.index.duplicated(keep="last")]

            data["Close"] = pd.to_numeric(
                data["Close"],
                errors="coerce"
            )

            data = data.dropna(subset=["Close"])
            data = data[data["Close"] > 0]

            if data.empty:
                continue

            dates = pd.to_datetime(data.index).date

            for item in records:
                try:
                    snapshot_date = pd.Timestamp(
                        item["date"]
                    ).date()

                    start_price = float(item["price"])

                    if start_price <= 0:
                        continue

                    future_mask = [
                        date > snapshot_date
                        for date in dates
                    ]

                    future = data.loc[future_mask]

                    for horizon in HORIZONS:
                        key = f"{horizon}_sessions"

                        if key in item["forward_returns"]:
                            continue

                        if len(future) < horizon:
                            continue

                        future_price = float(
                            future["Close"].iloc[horizon - 1]
                        )

                        change = (
                            future_price / start_price - 1
                        ) * 100

                        item["forward_returns"][key] = round(
                            change, 2
                        )

                        print(
                            f"{ticker}: {horizon} sesji = "
                            f"{change:+.2f}%"
                        )

                except (ValueError, TypeError, KeyError) as error:
                    print(
                        f"Blad pojedynczego wpisu {ticker}: {error}"
                    )

        except Exception as error:
            print(f"Blad historii {ticker}: {error}")

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

                if key not in returns:
                    continue

                try:
                    value = float(returns[key])
                    values.append(value)

                except (TypeError, ValueError):
                    continue

            if not values:
                statistics[signal][key] = {
                    "count": 0,
                    "average": None,
                    "median": None,
                    "positive": None
                }
                continue

            statistics[signal][key] = {
                "count": len(values),
                "average": round(
                    sum(values) / len(values), 2
                ),
                "median": round(
                    float(pd.Series(values).median()), 2
                ),
                "positive": round(
                    sum(value > 0 for value in values)
                    / len(values) * 100,
                    1
                )
            }

    return statistics


def print_statistics(statistics):
    print("\n" + "=" * 60)
    print("MARKET AI 11.0 - SKUTECZNOSC SYGNALOW")
    print("=" * 60)

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
                f"dodatnie wyniki: {values['positive']}% | "
                f"prob: {values['count']}"
            )


def update_performance(results):
    history = load_history()

    history = add_snapshots(history, results)
    history = evaluate_history(history)

    save_history(history)

    statistics = get_statistics(history)

    print_statistics(statistics)

    return statistics
