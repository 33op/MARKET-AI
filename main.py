
import os
import math
import json
import urllib.request
import urllib.error
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
VERSION = "16.2"

VIRTUAL_CAPITAL = 20000.00
RESERVE_AMOUNT = 2000.00
INVESTMENT_BUDGET = VIRTUAL_CAPITAL - RESERVE_AMOUNT

MAX_CANDIDATES_FOR_DIVERSIFICATION = 30
MAX_POSITIONS = 3

CORRELATION_PERIOD = "6mo"
HIGH_CORRELATION = 0.85
MEDIUM_CORRELATION = 0.70

DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")


# ============================================================
# PODSTAWOWE FUNKCJE
# ============================================================

def money(value):
    try:
        return (
            f"{float(value):,.2f} PLN"
            .replace(",", " ")
        )
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
        signal = str(
            result.get("signal", "")
        ).upper()

        if signal in summary:
            summary[signal] += 1

    return summary


# ============================================================
# FORMATOWANIE BACKTESTU v16.2
# ============================================================

def format_backtest_results(results):
    if not results:
        return "Brak wyników backtestu."

    if not isinstance(results, dict):
        return (
            "Nieprawidłowy format wyników backtestu. "
            "Oczekiwano raportu portfelowego."
        )

    lines = [
        f"**BACKTEST PORTFELOWY v{results.get('version', VERSION)}**",
        f"Instrumenty: {results.get('instruments', 0)}",
        f"Liczba sesji: {results.get('calendar_sessions', 0)}",
        ""
    ]

    test_start = results.get("test_start")
    test_end = results.get("test_end")

    if test_start is not None and test_end is not None:
        lines.append(
            f"Okres testowy: {test_start.date()} - "
            f"{test_end.date()}"
        )

    comparison = results.get("holding_comparison", {})

    if not comparison:
        lines.append("Brak danych porównawczych.")
        return "\n".join(lines)

    lines.append("")
    lines.append("**PORÓWNANIE CZASU TRZYMANIA POZYCJI**")

    for holding in [1, 3, 5]:
        item = comparison.get(holding)

        if item is None:
            item = comparison.get(str(holding))

        if not item:
            continue

        train = item.get("train", {})
        test = item.get("out_of_sample", {})

        lines.extend([
            "",
            f"**TRZYMANIE: {holding} SESJI**",
            (
                f"Trening: "
                f"{safe_float(train.get('return')):+.2f}% | "
                f"Kapitał: "
                f"{money(train.get('ending_capital', 0))}"
            ),
            (
                f"Test OOS: "
                f"{safe_float(test.get('return')):+.2f}% | "
                f"Kapitał: "
                f"{money(test.get('ending_capital', 0))}"
            ),
            (
                f"Wynik brutto OOS: "
                f"{safe_float(test.get('gross_return')):+.2f}%"
            ),
            (
                f"Obsunięcie OOS: "
                f"{safe_float(test.get('drawdown')):.2f}%"
            ),
            (
                f"Transakcje: {test.get('trades', 0)} | "
                f"Skuteczność: "
                f"{safe_float(test.get('win_rate')):.1f}%"
            ),
            (
                f"Średnia transakcja: "
                f"{safe_float(test.get('average_trade')):+.2f} PLN"
            )
        ])

        profit_factor = safe_float(
            test.get("profit_factor")
        )

        lines.append(
            f"Profit factor: {profit_factor:.2f}"
        )

    benchmark = results.get("benchmark")

    lines.extend([
        "",
        "**PORÓWNANIE Z WIG**",
        (
            f"WIG: {benchmark:+.2f}%"
            if benchmark is not None
            else "WIG: brak danych"
        )
    ])

    selected_holding = results.get(
        "holding_sessions",
        3
    )

    lines.extend([
        "",
        f"Domyślny wariant raportu: {selected_holding} sesje.",
        (
            f"Prowizja: "
            f"{safe_float(results.get('commission_rate')) * 100:.2f}%"
        ),
        (
            f"Poślizg: "
            f"{safe_float(results.get('slippage_rate')) * 100:.2f}%"
        ),
        "",
        "Wyniki historyczne nie gwarantują przyszłych wyników.",
        "Symulacja edukacyjna SIGG, bez składania rzeczywistych zleceń."
    ])

    return "\n".join(lines)


# ============================================================
# KANDYDACI DO PORTFELA
# ============================================================

def get_candidates(results):
    candidates = []

    for result in results:
        signal = str(
            result.get("signal", "")
        ).upper()

        score = safe_float(
            result.get("score")
        )

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

    return candidates[
        :MAX_CANDIDATES_FOR_DIVERSIFICATION
    ]


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

        if data is None or data.empty:
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):

            level0 = data.columns.get_level_values(0)
            level1 = data.columns.get_level_values(1)

            if "Close" in level0:
                prices = data.xs(
                    "Close",
                    axis=1,
                    level=0
                )

            elif "Close" in level1:
                prices = data.xs(
                    "Close",
                    axis=1,
                    level=1
                )

            else:
                return pd.DataFrame()

        else:
            if "Close" not in data.columns:
                return pd.DataFrame()

            prices = data[["Close"]].copy()

            if len(tickers) == 1:
                prices.columns = tickers

        if isinstance(prices, pd.Series):
            prices = prices.to_frame()

        prices.index = pd.to_datetime(prices.index)

        if getattr(prices.index, "tz", None) is not None:
            prices.index = prices.index.tz_localize(None)

        prices.index = prices.index.normalize()

        prices = prices[
            ~prices.index.duplicated(keep="last")
        ]

        prices = prices.sort_index()

        for column in prices.columns:
            prices[column] = pd.to_numeric(
                prices[column],
                errors="coerce"
            )

        prices = prices.dropna(
            axis=1,
            how="all"
        )

        # Nie usuwamy całych sesji tylko dlatego,
        # że jedna spółka nie miała notowań.
        return prices

    except Exception as error:
        print(
            f"Błąd pobierania korelacji: {error}"
        )

        return pd.DataFrame()


def calculate_correlation_matrix(tickers):
    if len(tickers) < 2:
        return pd.DataFrame()

    prices = download_correlation_data(tickers)

    if prices.empty:
        return pd.DataFrame()

    returns = prices.pct_change(fill_method=None)

    # Korelacja liczona parami.
    # Brak notowań jednej spółki nie usuwa danych
    # pozostałych instrumentów.
    matrix = returns.corr(
        min_periods=30
    )

    return matrix


def get_correlation(matrix, ticker_a, ticker_b):
    try:
        if matrix is None or matrix.empty:
            return None

        value = matrix.loc[
            ticker_a,
            ticker_b
        ]

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
        return [], [], ["Brak odpowiednich kandydatów."]

    tickers = [
        result.get("ticker")
        for result in candidates
        if result.get("ticker")
    ]

    matrix = calculate_correlation_matrix(tickers)

    selected = []
    selected_sectors = set()

    details = []
    rejection_reasons = []

    for candidate in candidates:

        if len(selected) >= MAX_POSITIONS:
            break

        ticker = candidate.get("ticker")

        if not ticker:
            rejection_reasons.append(
                "Kandydat bez symbolu."
            )
            continue

        sector = get_sector(ticker)

        if not sector:
            sector = "NIEZWERYFIKOWANY"

        # Znany sektor może być użyty tylko raz.
        # Nieznany sektor nie blokuje wyboru.
        if (
            sector != "NIEZWERYFIKOWANY"
            and sector in selected_sectors
        ):
            rejection_reasons.append(
                f"{ticker}: sektor {sector} "
                f"jest już reprezentowany."
            )
            continue

        acceptable = True
        correlations = []
        warnings = []

        for chosen in selected:
            chosen_ticker = chosen.get("ticker")

            correlation = get_correlation(
                matrix,
                ticker,
                chosen_ticker
            )

            if correlation is None:
                warnings.append(
                    f"Brak danych korelacji z {chosen_ticker}."
                )
                continue

            correlations.append(
                (chosen_ticker, correlation)
            )

            if abs(correlation) >= HIGH_CORRELATION:
                acceptable = False

                rejection_reasons.append(
                    f"{ticker}: zbyt wysoka korelacja "
                    f"z {chosen_ticker} "
                    f"({correlation:+.2f})."
                )

                break

            if abs(correlation) >= MEDIUM_CORRELATION:
                warnings.append(
                    f"Podwyższona korelacja z "
                    f"{chosen_ticker}: {correlation:+.2f}."
                )

        if not acceptable:
            continue

        selected.append(candidate)

        if sector != "NIEZWERYFIKOWANY":
            selected_sectors.add(sector)

        details.append({
            "ticker": ticker,
            "sector": sector,
            "correlations": correlations,
            "warnings": warnings,
            "correlation_data_available": bool(correlations)
        })

        if sector == "NIEZWERYFIKOWANY":
            rejection_reasons.append(
                f"{ticker}: wybrano mimo nieznanego sektora."
            )

        for warning in warnings:
            rejection_reasons.append(
                f"{ticker}: {warning}"
            )

    if not selected:
        return (
            [],
            [],
            rejection_reasons or [
                "Nie znaleziono odpowiednich kandydatów."
            ]
        )

    return selected, details, rejection_reasons


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

    budget_per_position = (
        INVESTMENT_BUDGET / len(selected)
    )

    allocations = []
    invested = 0.0

    for result in selected:
        ticker = result.get("ticker")
        price = safe_float(
            result.get("price")
        )

        if not ticker or price <= 0:
            continue

        shares = math.floor(
            budget_per_position / price
        )

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

    print(
        f"Kapitał początkowy: "
        f"{money(VIRTUAL_CAPITAL)}"
    )

    print(
        f"Zainwestowano:      "
        f"{money(summary['invested'])}"
    )

    print(
        f"Pozostała gotówka:  "
        f"{money(summary['remaining'])}"
    )

    print(
        f"Rezerwa planowana:  "
        f"{money(summary['reserve'])}"
    )

    print("=" * 65)


# ============================================================
# FORMATOWANIE DYWERSYFIKACJI
# ============================================================

def format_selection_details(details, rejection_reasons):
    lines = []

    if details:
        lines.append("WYBRANI KANDYDACI:")

        for item in details:
            lines.append(
                f"{item['ticker']} | "
                f"Sektor: {item['sector']}"
            )

            if item["correlations"]:
                for other, correlation in item["correlations"]:
                    lines.append(
                        f"  Korelacja z {other}: "
                        f"{correlation:+.2f}"
                    )

            else:
                lines.append(
                    "  Brak wystarczających danych korelacji."
                )

            for warning in item.get("warnings", []):
                lines.append(
                    f"  UWAGA: {warning}"
                )

    if rejection_reasons:
        lines.append("")
        lines.append("UWAGI DOTYCZĄCE DOBORU:")

        for reason in rejection_reasons[:15]:
            lines.append(f"- {reason}")

    return "\n".join(lines) if lines else "Brak szczegółów."


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
    rejection_reasons,
    portfolio_statistics,
    portfolio_status,
    backtest_report
):
    if not DISCORD_WEBHOOK:
        print(
            "Brak DISCORD_WEBHOOK. "
            "Pomijam wysyłanie raportu."
        )
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

        confirmation = safe_float(
            result.get("confirmation_days")
        )

        change = safe_float(
            result.get("change")
        )

        top10_text.append(
            f"**{index}. {ticker}**\n"
            f"{score:.0f}/10 | {signal} | "
            f"Ryzyko: {risk} | "
            f"Potwierdzenie: {confirmation:.0f}/3 | "
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

    correlation_text = format_selection_details(
        selection_details,
        rejection_reasons
    )

    fields = [
        {
            "name": "TOP 10 RYNKU",
            "value": (
                "\n\n".join(top10_text)[:1024]
                if top10_text
                else "Brak danych."
            ),
            "inline": False
        },
        {
            "name": "WYBRANY PORTFEL TOP 3",
            "value": (
                "\n\n".join(selected_text)[:1024]
                if selected_text
                else "Brak wybranych pozycji."
            ),
            "inline": False
        },
        {
            "name": "DYWERSYFIKACJA I KORELACJE",
            "value": correlation_text[:1024],
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
            "name": "HISTORYCZNA SKUTECZNOŚĆ SYGNAŁÓW",
            "value": format_performance(
                performance_statistics
            )[:1024],
            "inline": False
        },
        {
            "name": "PORTFEL VS WIG",
            "value": (
                str(portfolio_status)
                + "\n\n"
                + format_portfolio_statistics(
                    portfolio_statistics
                )
            )[:1024],
            "inline": False
        },
        {
            "name": "BACKTEST PORTFELOWY v16.2",
            "value": (
                backtest_report[:1024]
                if backtest_report
                else "Brak danych backtestu."
            ),
            "inline": False
        }
    ]

    payload = {
        "username": "MARKET AI",
        "embeds": [
            {
                "title": (
                    f"{PROJECT_NAME} v{VERSION} | RAPORT"
                ),
                "description": (
                    f"Data analizy: "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Przeanalizowane instrumenty: "
                    f"{len(results)}"
                ),
                "color": 3447003,
                "fields": fields,
                "footer": {
                    "text": (
                        "MARKET AI | Wirtualna symulacja SIGG | "
                        "Wyniki historyczne nie gwarantują przyszłych"
                    )
                },
                "timestamp": datetime.now().astimezone().isoformat()
            }
        ]
    }

    try:
        request = urllib.request.Request(
            DISCORD_WEBHOOK,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "User-Agent": f"MARKET-AI/{VERSION}"
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
        print(
            f"Discord HTTP error: {error.code}"
        )

        print(
            error.read().decode(
                "utf-8",
                errors="replace"
            )
        )

    except urllib.error.URLError as error:
        print(
            f"Discord connection error: {error.reason}"
        )

    except Exception as error:
        print(
            f"Discord error: {error}"
        )


# ============================================================
# GŁÓWNA ANALIZA
# ============================================================

def start():
    print("=" * 65)
    print(f"{PROJECT_NAME} v{VERSION}")
    print(f"Data analizy: {datetime.now()}")
    print(f"Liczba instrumentów: {len(INSTRUMENTS)}")
    print("=" * 65)

    results = []
    failed = []

    print("\nROZPOCZYNAM ANALIZĘ RYNKU")

    for index, ticker in enumerate(
        INSTRUMENTS,
        start=1
    ):
        print(
            f"\n[{index}/{len(INSTRUMENTS)}] "
            f"ANALIZA: {ticker}"
        )

        try:
            result = analyze_market(ticker)

            if not result:
                failed.append(ticker)
                print("Brak poprawnego wyniku.")
                continue

            result["ticker"] = ticker

            if "score" not in result:
                failed.append(ticker)
                print("Brak punktacji.")
                continue

            results.append(result)

            print(
                f"Wynik: {result.get('score')}/10 | "
                f"Sygnał: {result.get('signal')} | "
                f"Ryzyko: {result.get('risk')} | "
                f"Potwierdzenie: "
                f"{result.get('confirmation_days', 0)}/3"
            )

        except Exception as error:
            failed.append(ticker)

            print(
                f"Błąd analizy {ticker}: {error}"
            )

    print("\n" + "=" * 65)
    print("KONIEC ANALIZY RYNKU")
    print("=" * 65)

    print(
        f"Poprawnie przeanalizowane: "
        f"{len(results)}/{len(INSTRUMENTS)}"
    )

    print(
        f"Nieudane analizy: {len(failed)}"
    )

    if not results:
        print(
            "Brak wyników. Zatrzymuję analizę."
        )
        return

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

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
            f"{result.get('confirmation_days', 0)}/3"
        )

    # --------------------------------------------------------
    # PODSUMOWANIE SYGNAŁÓW
    # --------------------------------------------------------

    signal_summary = get_signal_summary(results)

    print("\nPODSUMOWANIE SYGNAŁÓW")

    print(
        f"Pozytywne: "
        f"{signal_summary['POZYTYWNY']}"
    )

    print(
        f"Neutralne: "
        f"{signal_summary['NEUTRALNY']}"
    )

    print(
        f"Negatywne: "
        f"{signal_summary['NEGATYWNY']}"
    )

    # --------------------------------------------------------
    # KANDYDACI
    # --------------------------------------------------------

    candidates = get_candidates(results)

    print(
        f"\nKandydaci do portfela: "
        f"{len(candidates)}"
    )

    for candidate in candidates:
        print(
            f"- {candidate['ticker']} | "
            f"{candidate['score']}/10 | "
            f"{candidate['signal']}"
        )

    # --------------------------------------------------------
    # DYWERSYFIKACJA
    # --------------------------------------------------------

    print("\nWYBÓR TOP 3")

    (
        selected,
        selection_details,
        rejection_reasons
    ) = select_diversified_candidates(candidates)

    if selected:
        for result in selected:
            print(
                f"{result.get('ticker')} | "
                f"{result.get('score')}/10 | "
                f"Sektor: "
                f"{get_sector(result.get('ticker'))}"
            )

    else:
        print(
            "Nie znaleziono kandydatów."
        )

    print("\nSZCZEGÓŁY DYWERSYFIKACJI")

    print(
        format_selection_details(
            selection_details,
            rejection_reasons
        )
    )

    # --------------------------------------------------------
    # ALOKACJA WIRTUALNEGO KAPITAŁU
    # --------------------------------------------------------

    allocations, summary = allocate_portfolio(
        selected
    )

    print_portfolio(
        allocations,
        summary
    )

    # --------------------------------------------------------
    # HISTORIA SKUTECZNOŚCI SYGNAŁÓW
    # --------------------------------------------------------

    print(
        "\nAKTUALIZACJA HISTORII SYGNAŁÓW"
    )

    try:
        performance_statistics = update_performance(
            results
        )

    except Exception as error:
        print(
            f"Błąd historii skuteczności: {error}"
        )

        performance_statistics = {}

    # --------------------------------------------------------
    # HISTORIA PORTFELA I WIG
    # --------------------------------------------------------

    print(
        "\nAKTUALIZACJA HISTORII PORTFELA"
    )

    try:
        (
            portfolio_statistics,
            portfolio_status
        ) = update_portfolio_history(
            allocations,
            summary,
            results
        )

    except Exception as error:
        print(
            f"Błąd historii portfela: {error}"
        )

        portfolio_statistics = {}

        portfolio_status = (
            "Nie udało się zaktualizować historii portfela."
        )

    print("\nSTATYSTYKI PORTFELA VS WIG")

    print(portfolio_status)

    print(
        format_portfolio_statistics(
            portfolio_statistics
        )
    )

    # --------------------------------------------------------
    # BACKTEST PORTFELOWY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("URUCHAMIANIE BACKTESTU HISTORYCZNEGO")
    print("=" * 65)

    try:
        backtest_results = run_backtest(
            INSTRUMENTS
        )

        backtest_report = format_backtest_results(
            backtest_results
        )

        print("\nPODSUMOWANIE BACKTESTU")
        print(backtest_report)

    except Exception as error:
        print(
            f"Błąd backtestu: {error}"
        )

        backtest_report = (
            "Nie udało się wykonać backtestu."
        )

    # --------------------------------------------------------
    # RAPORT DISCORD
    # --------------------------------------------------------

    print("\nWYSYŁANIE RAPORTU DISCORD")

    send_discord_report(
        results,
        selected,
        allocations,
        summary,
        signal_summary,
        performance_statistics,
        selection_details,
        rejection_reasons,
        portfolio_statistics,
        portfolio_status,
        backtest_report
    )

    print("\n" + "=" * 65)
    print(
        f"{PROJECT_NAME} v{VERSION} | ZAKOŃCZONO"
    )
    print("=" * 65)


if __name__ == "__main__":
    start()
