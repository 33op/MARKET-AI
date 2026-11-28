import json
import os
import math
from collections import defaultdict

import yfinance as yf
import pandas as pd


VERSION = "14.0"
EVALUATION_VERSION = 3

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "performance.json")

HORIZONS = [1, 5, 10]
SIGNALS = ["POZYTYWNY", "NEUTRALNY", "NEGATYWNY"]
HISTORY_PERIOD = "2y"

# Sygnał powstaje po zamknięciu sesji, dlatego wejście następuje
# najwcześniej po cenie OPEN kolejnej sesji.
MAX_PRICE_DIFFERENCE_PCT = 1.0

# Powtarzające się sygnały tej samej spółki w krótkim odstępie
# traktujemy jako jeden epizod.
EPISODE_GAP_CALENDAR_DAYS = 5


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

    except (json.JSONDecodeError, OSError) as error:
        print(f"Nie mozna odczytac historii analiz: {error}")
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
        (str(item.get("ticker")), str(item.get("date")))
        for item in history
        if isinstance(item, dict)
    }

    added = 0

    for result in results:
        if not isinstance(result, dict):
            continue

        ticker = result.get("ticker")
        date = result.get("date")
        price = result.get("price")

        if not ticker or not date or not valid_number(price):
            continue

        price = float(price)
        if price <= 0:
            continue

        try:
            normalized_date = str(pd.Timestamp(date).date())
        except (TypeError, ValueError):
            continue

        key = (str(ticker), normalized_date)
        if key in existing:
            continue

        signal = result.get("signal", "NEUTRALNY")
        if signal not in SIGNALS:
            signal = "NEUTRALNY"

        score = result.get("score", 0)
        if not valid_number(score):
            score = 0

        confirmed_days = result.get("confirmed_days", 0)
        if not valid_number(confirmed_days):
            confirmed_days = 0

        history.append({
            "ticker": str(ticker),
            "date": normalized_date,
            "price": price,
            "score": float(score),
            "signal": signal,
            "risk": str(result.get("risk", "NIEZNANE")),
            "confirmed_days": int(confirmed_days),
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

    if data is None or data.empty:
        return None

    if not {"Open", "Close"}.issubset(data.columns):
        return None

    data = data.copy().sort_index()
    data = data[~data.index.duplicated(keep="last")]

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    for column in ("Open", "Close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=["Open", "Close"])
    data = data[
        (data["Open"] > 0)
        & (data["Close"] > 0)
        & data["Open"].map(lambda value: math.isfinite(float(value)))
        & data["Close"].map(lambda value: math.isfinite(float(value)))
    ]

    if data.empty:
        return None

    data["SessionDate"] = [timestamp.date() for timestamp in data.index]
    return data


def evaluate_ticker_records(ticker, records):
    print(f"\nUZUPELNIANIE HISTORII: {ticker}")

    try:
        data = prepare_market_data(ticker)

        if data is None:
            print(f"Brak poprawnych danych OHLC: {ticker}")
            return

        dates = list(data["SessionDate"])
        closes = list(data["Close"])
        opens = list(data["Open"])
        position_by_date = {
            session_date: index
            for index, session_date in enumerate(dates)
        }

        for item in records:
            try:
                snapshot_date = pd.Timestamp(item["date"]).date()

                if snapshot_date not in position_by_date:
                    print(f"{ticker}: brak sesji dla {snapshot_date}")
                    continue

                stored_price = float(item["price"])
                if not valid_number(stored_price) or stored_price <= 0:
                    continue

                snapshot_position = position_by_date[snapshot_date]
                actual_close = float(closes[snapshot_position])

                difference = abs(actual_close / stored_price - 1) * 100
                if difference > MAX_PRICE_DIFFERENCE_PCT:
                    print(
                        f"{ticker}: OSTRZEZENIE - cena zapisana rozni sie "
                        f"od Yahoo o {difference:.2f}%"
                    )
                    continue

                entry_position = snapshot_position + 1
                if entry_position >= len(data):
                    continue

                entry_price = float(opens[entry_position])
                if not valid_number(entry_price) or entry_price <= 0:
                    continue

                forward_returns = item.setdefault("forward_returns", {})
                if not isinstance(forward_returns, dict):
                    forward_returns = {}
                    item["forward_returns"] = forward_returns

                for horizon in HORIZONS:
                    key = f"{horizon}_sessions"

                    if key in forward_returns:
                        continue

                    exit_position = entry_position + horizon - 1
                    if exit_position >= len(data):
                        continue

                    exit_price = float(closes[exit_position])
                    if not valid_number(exit_price) or exit_price <= 0:
                        continue

                    change = (exit_price / entry_price - 1) * 100
                    if not math.isfinite(change):
                        continue

                    forward_returns[key] = round(change, 2)
                    item.setdefault("evaluation_details", {})[key] = {
                        "entry_date": str(dates[entry_position]),
                        "entry_price": round(entry_price, 6),
                        "exit_date": str(dates[exit_position]),
                        "exit_price": round(exit_price, 6),
                        "method": "NEXT_SESSION_OPEN_TO_HORIZON_CLOSE"
                    }

                    print(
                        f"{ticker}: {horizon} sesji "
                        f"({dates[entry_position]} -> {dates[exit_position]}) "
                        f"= {change:+.2f}%"
                    )

            except (ValueError, TypeError, KeyError, IndexError) as error:
                print(f"Blad pojedynczego wpisu {ticker}: {error}")

    except Exception as error:
        print(f"Blad historii {ticker}: {error}")


def evaluate_history(history):
    grouped = defaultdict(list)

    for item in history:
        if not isinstance(item, dict):
            continue

        if not all(key in item for key in ("ticker", "date", "price")):
            continue

        if not valid_number(item.get("price")) or float(item["price"]) <= 0:
            continue

        if not isinstance(item.get("forward_returns"), dict):
            item["forward_returns"] = {}

        # Po zmianie metodologii przeliczamy wszystkie historyczne wyniki.
        # Same snapshoty (data, cena i sygnal) zostaja zachowane.
        if item.get("evaluation_version", 0) != EVALUATION_VERSION:
            item["forward_returns"] = {}
            item.pop("evaluation_details", None)
            item["evaluation_version"] = EVALUATION_VERSION

        if len(item["forward_returns"]) < len(HORIZONS):
            grouped[str(item["ticker"])].append(item)

    for ticker, records in grouped.items():
        evaluate_ticker_records(ticker, records)

    return history


def select_independent_episodes(records):
    """
    Dla kazdej spolki dzieli historie na epizody sygnalow.
    Epizod konczy sie po zmianie sygnalu albo przerwie dluzszej
    niz EPISODE_GAP_CALENDAR_DAYS.

    Dla kazdego horyzontu wybiera pierwszy dostepny wynik w epizodzie.
    Dzięki temu brak danych dla najstarszego snapshotu nie usuwa
    automatycznie pozniejszych, kompletnych obserwacji z tego epizodu.
    """
    by_ticker = defaultdict(list)

    for item in records:
        if not isinstance(item, dict):
            continue
        if item.get("signal") not in SIGNALS:
            continue
        if not isinstance(item.get("forward_returns"), dict):
            continue

        by_ticker[str(item.get("ticker", ""))].append(item)

    selected_by_horizon = {
        horizon: [] for horizon in HORIZONS
    }

    for ticker_records in by_ticker.values():
        ticker_records.sort(key=lambda item: str(item.get("date", "")))

        episodes = []
        current_episode = []
        previous_signal = None
        previous_date = None

        for item in ticker_records:
            try:
                current_date = pd.Timestamp(item["date"]).date()
            except (KeyError, TypeError, ValueError):
                continue

            current_signal = item.get("signal")

            is_new_episode = (
                previous_date is None
                or current_signal != previous_signal
                or (current_date - previous_date).days
                > EPISODE_GAP_CALENDAR_DAYS
            )

            if is_new_episode:
                if current_episode:
                    episodes.append(current_episode)
                current_episode = []

            current_episode.append(item)
            previous_signal = current_signal
            previous_date = current_date

        if current_episode:
            episodes.append(current_episode)

        for episode in episodes:
            for horizon in HORIZONS:
                key = f"{horizon}_sessions"

                for item in episode:
                    value = item.get("forward_returns", {}).get(key)
                    if valid_number(value):
                        selected_by_horizon[horizon].append(item)
                        break

    return selected_by_horizon


def get_statistics(history):
    statistics = {}
    independent_records = select_independent_episodes(history)

    for signal in SIGNALS:
        statistics[signal] = {}

        for horizon in HORIZONS:
            key = f"{horizon}_sessions"
            values = []

            for item in independent_records[horizon]:
                if item.get("signal") != signal:
                    continue

                value = item.get("forward_returns", {}).get(key)
                if valid_number(value):
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
                "average": round(sum(values) / len(values), 2),
                "median": round(float(pd.Series(values).median()), 2),
                "positive": round(
                    sum(value > 0 for value in values) / len(values) * 100,
                    1
                ),
                "negative": round(
                    sum(value < 0 for value in values) / len(values) * 100,
                    1
                )
            }

    return statistics


def print_statistics(statistics):
    print("\n" + "=" * 70)
    print(f"MARKET AI {VERSION} - SKUTECZNOSC SYGNALOW")
    print("=" * 70)
    print("Metoda: nastepna sesja OPEN -> CLOSE po zadanym horyzoncie.")
    print("Statystyki ograniczaja powtarzajace sie snapshoty w jednym epizodzie.")

    for signal in SIGNALS:
        print(f"\nSYGNAL: {signal}")

        for horizon in HORIZONS:
            key = f"{horizon}_sessions"
            values = statistics[signal][key]

            if values["count"] == 0:
                print(f"{key}: za malo danych")
                continue

            print(
                f"{key}: "
                f"srednia: {values['average']:+.2f}% | "
                f"mediana: {values['median']:+.2f}% | "
                f"dodatnie: {values['positive']}% | "
                f"ujemne: {values['negative']}% | "
                f"niezalezne epizody: {values['count']}"
            )

    print("\n" + "=" * 70)
    print("UWAGA:")
    print("Wyniki historyczne nie gwarantuja przyszlych rezultatow.")
    print(
        "Ograniczenie epizodow zmniejsza nakladanie obserwacji, "
        "ale nie zapewnia pelnej niezaleznosci statystycznej."
    )
    print(
        "Wynik od nastepnego OPEN do przyszlego CLOSE nie uwzglednia "
        "prowizji, poslizgu ani ograniczen plynnosci."
    )
    print("=" * 70)


def update_performance(results):
    history = load_history()
    history = add_snapshots(history, results)
    history = evaluate_history(history)
    save_history(history)

    statistics = get_statistics(history)
    print_statistics(statistics)

    return statistics
