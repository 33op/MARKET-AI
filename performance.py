
import json
import os
from collections import defaultdict

import yfinance as yf
import pandas as pd

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "performance.json")

HORIZONS = [1, 5, 10]


def load_history():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        print("Nie mozna odczytac historii analiz.")
        return []


def save_history(history):
    os.makedirs(DATA_DIR, exist_ok=True)

    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=4)


def add_snapshots(history, results):
    existing = {
        (item["ticker"], item["date"])
        for item in history
    }

    added = 0

    for result in results:
        key = (result["ticker"], result["date"])

        if key in existing:
            continue

        history.append({
            "ticker": result["ticker"],
            "date": result["date"],
            "price": result["price"],
            "score": result["score"],
            "signal": result["signal"],
            "risk": result["risk"],
            "forward_returns": {}
        })

        existing.add(key)
        added += 1

    print(f"\nZapisano nowych analiz: {added}")

    return history


def evaluate_history(history):
    grouped = defaultdict(list)

    for item in history:
        if len(item["forward_returns"]) < len(HORIZONS):
            grouped[item["ticker"]].append(item)

    for ticker, records in grouped.items():
        try:
            data = yf.Ticker(ticker).history(
                period="1y",
                interval="1d",
                auto_adjust=True
            )

            if data.empty:
                continue

            data = data.dropna(subset=["Close"])

            dates = pd.to_datetime(data.index).date

            for item in records:
                snapshot_date = pd.Timestamp(item["date"]).date()

                future = data.loc[
                    [date > snapshot_date for date in dates]
                ]

                for horizon in HORIZONS:
                    key = f"{horizon}_sessions"

                    if key in item["forward_returns"]:
                        continue

                    if len(future) < horizon:
                        continue

                    future_price = float(
                        future["Close"].iloc[horizon - 1]
                    )

                    start_price = float(item["price"])

                    if start_price <= 0:
                        continue

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

        except Exception as error:
            print(f"Blad historii {ticker}: {error}")

    return history


def get_statistics(history):
    statistics = {}

    for signal in ["POZYTYWNY", "NEUTRALNY", "NEGATYWNY"]:
        statistics[signal] = {}

        for horizon in HORIZONS:
            key = f"{horizon}_sessions"

            values = [
                item["forward_returns"][key]
                for item in history
                if item["signal"] == signal
                and key in item["forward_returns"]
            ]

            if not values:
                statistics[signal][key] = {
                    "count": 0,
                    "average": None,
                    "positive": None
                }
                continue

            statistics[signal][key] = {
                "count": len(values),
                "average": round(
                    sum(values) / len(values), 2
                ),
                "positive": round(
                    sum(value > 0 for value in values)
                    / len(values) * 100,
                    1
                )
            }

    return statistics


def update_performance(results):
    history = load_history()

    history = add_snapshots(history, results)
    history = evaluate_history(history)

    save_history(history)

    statistics = get_statistics(history)

    print("\n" + "=" * 55)
    print("MARKET AI 9.0 - SKUTECZNOSC ANALIZ")
    print("=" * 55)

    for signal, horizons in statistics.items():
        print(f"\nSYGNAL: {signal}")

        for horizon, values in horizons.items():
            if values["count"] == 0:
                print(f"{horizon}: za malo danych")
            else:
                print(
                    f"{horizon}: "
                    f"srednia zmiana {values['average']:+.2f}% | "
                    f"dodatni wynik: {values['positive']}% | "
                    f"prob: {values['count']}"
                )

    return statistics
