
import os
import math
import urllib.request
import urllib.error
import json
from datetime import datetime

import yfinance as yf
import pandas as pd

from market_data import analyze_market
from instruments import INSTRUMENTS
from performance import update_performance
from sectors import get_sector
from backtest import run_backtest
from portfolio import (
    update_portfolio_history,
    format_statistics as format_portfolio_statistics
)


# ============================================================
# KONFIGURACJA
# ============================================================

PROJECT_NAME = "MARKET AI"
VERSION = "14.0"

VIRTUAL_CAPITAL = 20000.00
RESERVE_AMOUNT = 2000.00
INVESTMENT_BUDGET = VIRTUAL_CAPITAL - RESERVE_AMOUNT

MAX_CANDIDATES_FOR_DIVERSIFICATION = 30

CORRELATION_PERIOD = "6mo"
HIGH_CORRELATION = 0.85
MEDIUM_CORRELATION = 0.70

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")


# ============================================================
# PODSTAWOWE FUNKCJE
# ============================================================

def money(value):
    try:
        return f"{float(value):,.2f} PLN".replace(",", " ")
    except (TypeError, ValueError):
        return "BRAK DANYCH"


def safe_float(value, default=0.0):
    try:
        number = float(value)

        if math.isfinite(number):
            return number

    except (TypeError, ValueError):
        pass

    return default


def ranking_key(result):
    return (
        safe_float(result.get("score")),
        safe_float(result.get("confirmation_days")),
        safe_float(result.get("momentum_20d")),
        safe_float(result.get("momentum_5d"))
    )


def format_performance(statistics):
    if not statistics:
        return "Brak danych historycznych."

    lines = []

    for signal, horizons in statistics.items():
        lines.append(f"**{signal.upper()}**")

        for horizon, values in horizons.items():
            count = values.get("count", 0)

            if count == 0:
                continue

            average = safe_float(values.get("average"))
            median = safe_float(values.get("median"))
            positive = safe_float(values.get("positive"))

            lines.append(
                f"{horizon}: {average:+.2f}% średnio | "
                f"mediana {median:+.2f}% | "
                f"dodatnie {positive:.1f}% | "
                f"próby: {count}"
            )

    return "\n".join(lines) if lines else "Za mało danych."


def get_signal_summary(results):
    summary = {
        "POZYTYWNY": 0,
        "NEUTRALNY": 0,
        "NEGATYWNY": 0
    }

    for result in results:
        signal = str(result.get("signal", "")).upper()

        if signal in summary:
            summary[signal] += 1

    return summary


# ============================================================
# FORMATOWANIE BACKTESTU
# ============================================================

def format_backtest_results(results):
    if not results:
        return "Brak wyników backtestu."

    valid = [
        item for item in results
        if item.get("average_return") is not None
        and item.get("positive_count", 0) > 0
        and item.get("win_rate") is not None
    ]

    if not valid:
        return "Brak wystarczających danych."

    total_signals = sum(
        item["positive_count"]
        for item in valid
    )

    if total_signals == 0:
        return "Brak pozytywnych sygnałów."

    weighted_return = sum(
        item["average_return"] * item["positive_count"]
        for item in valid
    ) / total_signals

    total_wins = sum(
        item["win_rate"] / 100 * item["positive_count"]
        for item in valid
    )

    win_rate = total_wins / total_signals * 100

    top = sorted(
        valid,
        key=lambda item: item["average_return"],
        reverse=True
    )[:5]

    lines = [
        "**TEST HISTORYCZNY SYGNAŁÓW**",
        f"Instrumenty: {len(results)}",
        f"Pozytywne sygnały: {total_signals}",
        f"Średni wynik: {weighted_return:+.2f}%",
        f"Udział dodatnich wyników: {win_rate:.1f}%",
        "",
        "**TOP 5 BACKTESTU**"
    ]

    for item in top:
        lines.append(
            f"{item['ticker']}: "
            f"{item['average_return']:+.2f}% | "
            f"Skuteczność: {item['win_rate']:.1f}%"
        )

    lines.extend([
        "",
        "Pomiar jednodniowy, bez kosztów transakcyjnych.",
        "To nie jest wynik pełnego portfela."
    ])

    return "\n".join(lines)


# ============================================================
# KANDYDACI DO PORTFELA
# ============================================================

def get_candidates(results):
    candidates = []

    for result in results:
        signal = str(result.get("signal", "")).upper()

        score = safe_float(result.get("score"))
        confirmation = safe_float(
            result.get("confirmation_days")
        )

        if (
            signal == "POZYTYWNY"
            and score >= 7
            and confirmation >= 2
        ):
            candidates.append(result)

    candidates.sort(
        key=ranking_key,
        reverse=True
    )

    return candidates[:MAX_CANDIDATES_FOR_DIVERSIFICATION]


# ============================================================
# KORELACJE
# ============================================================

def download_correlation_data(tickers):
    if not tickers:
        return pd.DataFrame()

    try:
        data = yf.download(
            tickers=tickers,
            period=CORRELATION_PERIOD,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True
        )

        if data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            if "Close" in data.columns.get_level_values(0):
                prices = data["Close"]
            else:
                return pd.DataFrame()

        else:
            if "Close" not in data.columns:
                return pd.DataFrame()

            prices = data["Close"]

        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        prices = prices.dropna(axis=1, how="all")
        prices = prices.dropna()

        return prices

    except Exception as error:
        print(f"Błąd pobierania korelacji: {error}")
        return pd.DataFrame()


def calculate_correlation_matrix(tickers):
    if len(tickers) < 2:
        return pd.DataFrame()

    prices = download_correlation_data(tickers)

    if prices.empty:
        return pd.DataFrame()

    returns = prices.pct_change().dropna()

    if returns.empty:
        return pd.DataFrame()

    return returns.corr()


def get_correlation(matrix, ticker_a, ticker_b):
    try:
        value = matrix.loc[ticker_a, ticker_b]

        if pd.isna(value):
            return None

        return float(value)

    except Exception:
        return None


# ============================================================
# DYWERSYFIKACJA TOP 3
# ============================================================

def select_diversified_candidates(candidates):
    if not candidates:
        return [], "Brak odpowiednich kandydatów."

    tickers = [
        result.get("ticker")
        for result in candidates
        if result.get("ticker")
    ]

    matrix = calculate_correlation_matrix(tickers)

    selected = []
    selected_sectors = set()
    details = []

    for candidate in candidates:
        if len(selected) >= 3:
            break

        ticker = candidate.get("ticker")

        if not ticker:
            continue

        sector = get_sector(ticker)

        if not sector or sector == "NIEZWERYFIKOWANY":
            continue

        if sector in selected_sectors:
            continue

        acceptable = True
        correlations = []

        for chosen in selected:
            chosen_ticker = chosen.get("ticker")

            correlation = get_correlation(
                matrix,
                ticker,
                chosen_ticker
            )

            if correlation is None:
                acceptable = False
                break

            correlations.append(
                (chosen_ticker, correlation)
            )

            if abs(correlation) >= HIGH_CORRELATION:
                acceptable = False
                break

        if not acceptable:
            continue

        selected.append(candidate)
        selected_sectors.add(sector)

        details.append({
            "ticker": ticker,
            "sector": sector,
            "correlations": correlations
        })

    if not selected:
        return (
            [],
            "Nie znaleziono kandydatów spełniających "
            "warunki dywersyfikacji."
        )

    return selected, details


# ============================================================
# WIRTUALNA ALOKACJA
# ============================================================

def allocate_portfolio(selected):
    if not selected:
        return [], {
            "invested": 0.0,
            "remaining": VIRTUAL_CAPITAL,
            "reserve": RESERVE_AMOUNT
        }

    budget_per_position = INVESTMENT_BUDGET / len(selected)

    allocations = []
    invested = 0.0

    for result in selected:
        ticker = result.get("ticker")
        price = safe_float(result.get("price"))

        if not ticker or price <= 0:
            continue

        shares = math.floor(budget_per_position / price)

        if shares <= 0:
            continue

        cost = shares * price

        allocations.append({
            "ticker": ticker,
            "price": price,
            "shares": shares,
            "value": cost,
            "sector": get_sector(ticker)
        })

        invested += cost

    remaining = VIRTUAL_CAPITAL - invested

    summary = {
        "invested": invested,
        "remaining": remaining,
        "reserve": RESERVE_AMOUNT
    }

    return allocations, summary


# ============================================================
# PODSUMOWANIE PORTFELA
# ============================================================

def print_portfolio(allocations, summary):
    print("\n" + "=" * 65)
    print("WIRTUALNY PORTFEL MARKET AI")
    print("=" * 65)

    if not allocations:
        print("Brak wybranych pozycji.")

    else:
        for position in allocations:
            print(
                f"{position['ticker']} | "
                f"{position['shares']} szt. | "
                f"{position['price']:.2f} PLN | "
                f"Wartość: {money(position['value'])} | "
                f"Sektor: {position['sector']}"
            )

    print("-" * 65)
    print(f"Kapitał początkowy: {money(VIRTUAL_CAPITAL)}")
    print(f"Zainwestowano:      {money(summary['invested'])}")
    print(f"Pozostała gotówka:  {money(summary['remaining'])}")
    print(f"Rezerwa planowana:  {money(summary['reserve'])}")
    print("=" * 65)


# ============================================================
# RAPORT DISCORD
# ============================================================

def send_discord_report(
    results,
    selected,
    allocations,
    summary,
    signal_summary,
    performance_statistics,
    selection_details,
    portfolio_statistics,
    portfolio_status,
    backtest_report
):
    if not DISCORD_WEBHOOK:
        print("Brak DISCORD_WEBHOOK. Pomijam wysyłanie raportu.")
        return

    top10 = sorted(
        results,
        key=ranking_key,
        reverse=True
    )[:10]

    top10_text = []

    for index, result in enumerate(top10, start=1):
        ticker = result.get("ticker", "BRAK")
        signal = result.get("signal", "BRAK")
        score = safe_float(result.get("score"))
        risk = result.get("risk", "BRAK")
        confirmation = result.get("confirmation_days", 0)
        change = safe_float(result.get("daily_change"))

        top10_text.append(
            f"**{index}. {ticker}**\n"
            f"{score:.0f}/10 | {signal} | "
            f"Ryzyko: {risk} | "
            f"Potwierdzenie: {confirmation}/3 | "
            f"{change:+.2f}%"
        )

    selected_text = []

    for position in allocations:
        selected_text.append(
            f"**{position['ticker']}**\n"
            f"{position['shares']} szt. | "
            f"{money(position['value'])} | "
            f"{position['sector']}"
        )

    correlation_text = []

    if isinstance(selection_details, list):
        for item in selection_details:
            for other_ticker, correlation in item["correlations"]:
                correlation_text.append(
                    f"{item['ticker']} / {other_ticker}: "
                    f"{correlation:+.2f}"
                )

    fields = [
        {
            "name": "TOP 10 RYNKU",
            "value": "\n\n".join(top10_text)[:1024]
            if top10_text else "Brak danych.",
            "inline": False
        },
        {
            "name": "WYBRANY PORTFEL TOP 3",
            "value": "\n\n".join(selected_text)[:1024]
            if selected_text else "Brak wybranych pozycji.",
            "inline": False
        },
        {
            "name": "KORELACJE",
            "value": "\n".join(correlation_text)[:1024]
            if correlation_text else "Brak danych korelacji.",
            "inline": False
        },
        {
            "name": "KAPITAŁ WIRTUALNY",
            "value": (
                f"Kapitał: {money(VIRTUAL_CAPITAL)}\n"
                f"Zainwestowano: {money(summary['invested'])}\n"
                f"Gotówka: {money(summary['remaining'])}\n"
                f"Rezerwa planowana: {money(summary['reserve'])}"
            ),
            "inline": True
        },
        {
            "name": "SYGNAŁY RYNKOWE",
            "value": (
                f"Pozytywne: {signal_summary['POZYTYWNY']}\n"
                f"Neutralne: {signal_summary['NEUTRALNY']}\n"
                f"Negatywne: {signal_summary['NEGATYWNY']}"
            ),
            "inline": True
        },
        {
            "name": "HISTORYCZNA SKUTECZNOŚĆ",
            "value": format_performance(
                performance_statistics
            )[:1024],
            "inline": False
        },
        {
            "name": "PORTFEL VS WIG",
            "value": (
                portfolio_status
                + "\n\n"
                + format_portfolio_statistics(
                    portfolio_statistics
                )
            )[:1024],
            "inline": False
        },
        {
            "name": "BACKTEST HISTORYCZNY",
            "value": backtest_report[:1024],
            "inline": False
        }
    ]

    payload = {
        "embeds": [
            {
                "title": f"{PROJECT_NAME} v{VERSION} | RAPORT",
                "description": (
                    f"Data analizy: "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Przeanalizowane instrumenty: {len(results)}"
                ),
                "color": 3447003,
                "fields": fields,
                "footer": {
                    "text": (
                        "MARKET AI | Wirtualna symulacja SIGG | "
                        "Wyniki historyczne nie gwarantują przyszłych"
                    )
                }
            }
        ]
    }

    try:
        request = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": "MARKET-AI"
            },
            method="POST"
        )

        with urllib.request.urlopen(
            request,
            timeout=20
        ) as response:
            print(
                f"Discord: raport wysłany. "
                f"HTTP {response.status}"
            )

    except urllib.error.HTTPError as error:
        print(f"Discord HTTP error: {error.code}")
        print(error.read().decode("utf-8", errors="replace"))

    except Exception as error:
        print(f"Discord error: {error}")


# ============================================================
# GŁÓWNA ANALIZA
# ============================================================

def start():
    print("=" * 65)
    print(f"{PROJECT_NAME} v{VERSION}")
    print(f"Data analizy: {datetime.now()}")
    print("=" * 65)

    results = []

    print("\nROZPOCZYNAM ANALIZĘ RYNKU\n")

    for ticker in INSTRUMENTS:
        print("-" * 45)
        print(f"ANALIZA: {ticker}")

        try:
            result = analyze_market(ticker)

            if not result:
                print(f"Odrzucono {ticker}: brak wyniku.")
                continue

            result["ticker"] = ticker

            if "score" not in result:
                print(f"Odrzucono {ticker}: brak oceny.")
                continue

            results.append(result)

            print(
                f"Wynik: {result.get('score')}/10 | "
                f"Sygnał: {result.get('signal')} | "
                f"Ryzyko: {result.get('risk')}"
            )

        except Exception as error:
            print(f"Błąd analizy {ticker}: {error}")

    print("\n" + "=" * 65)
    print("KONIEC ANALIZY RYNKU")
    print("=" * 65)

    print(
        f"Poprawnie przeanalizowane instrumenty: "
        f"{len(results)}/{len(INSTRUMENTS)}"
    )

    if not results:
        print("Brak wyników. Zatrzymuję analizę.")
        return

    # Ranking
    ranked_results = sorted(
        results,
        key=ranking_key,
        reverse=True
    )

    print("\nTOP 10 RYNKU")

    for index, result in enumerate(
        ranked_results[:10],
        start=1
    ):
        print(
            f"{index}. {result.get('ticker')} | "
            f"{result.get('signal')} | "
            f"{result.get('score')}/10 | "
            f"RYZYKO: {result.get('risk')} | "
            f"Potwierdzenie: "
            f"{result.get('confirmation_days')}/3"
        )

    # Sygnały
    signal_summary = get_signal_summary(results)

    print("\nPODSUMOWANIE SYGNAŁÓW")
    print(f"Pozytywne: {signal_summary['POZYTYWNY']}")
    print(f"Neutralne: {signal_summary['NEUTRALNY']}")
    print(f"Negatywne: {signal_summary['NEGATYWNY']}")

    # Kandydaci
    candidates = get_candidates(results)

    print(
        f"\nKandydaci do portfela: {len(candidates)}"
    )

    # Dywersyfikacja
    selected, selection_details = (
        select_diversified_candidates(candidates)
    )

    print("\nWYBÓR TOP 3")

    if selected:
        for result in selected:
            print(
                f"{result.get('ticker')} | "
                f"{result.get('score')}/10 | "
                f"Sektor: {get_sector(result.get('ticker'))}"
            )

    else:
        print(selection_details)

    # Alokacja
    allocations, summary = allocate_portfolio(selected)

    print_portfolio(allocations, summary)

    # Historia skuteczności sygnałów
    print("\nAKTUALIZACJA HISTORII SYGNAŁÓW")

    try:
        performance_statistics = update_performance(results)

    except Exception as error:
        print(f"Błąd historii skuteczności: {error}")
        performance_statistics = {}

    # Historia portfela i WIG
    print("\nAKTUALIZACJA HISTORII PORTFELA")

    try:
        portfolio_statistics, portfolio_status = (
            update_portfolio_history(
                allocations,
                summary,
                results
            )
        )

    except Exception as error:
        print(f"Błąd historii portfela: {error}")
        portfolio_statistics = {}
        portfolio_status = (
            "Nie udało się zaktualizować historii portfela."
        )

    print("\nSTATYSTYKI PORTFELA VS WIG")
    print(portfolio_status)
    print(format_portfolio_statistics(portfolio_statistics))

    # Backtest
    print("\n" + "=" * 65)
    print("URUCHAMIANIE BACKTESTU HISTORYCZNEGO")
    print("=" * 65)

    try:
        backtest_results = run_backtest(INSTRUMENTS)

        backtest_report = format_backtest_results(
            backtest_results
        )

        print("\nPODSUMOWANIE BACKTESTU")
        print(backtest_report)

    except Exception as error:
        print(f"Błąd backtestu: {error}")
        backtest_report = (
            "Nie udało się wykonać backtestu."
        )

    # Discord
    print("\nWYSYŁANIE RAPORTU DISCORD")

    send_discord_report(
        results,
        selected,
        allocations,
        summary,
        signal_summary,
        performance_statistics,
        selection_details,
        portfolio_statistics,
        portfolio_status,
        backtest_report
    )

    print("\n" + "=" * 65)
    print(f"{PROJECT_NAME} v{VERSION} | ZAKOŃCZONO")
    print("=" * 65)


if __name__ == "__main__":
    start()
