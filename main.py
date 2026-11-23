import os
import json
import math
import urllib.request
import urllib.error
from datetime import datetime

import yfinance as yf
import pandas as pd

from market_data import analyze_market
from instruments import INSTRUMENTS
from performance import update_performance
from sectors import get_sector

PROJECT_NAME = "MARKET AI"
VERSION = "12.0"

VIRTUAL_CAPITAL = 20000.00
RESERVE_AMOUNT = 2000.00
INVESTMENT_BUDGET = VIRTUAL_CAPITAL - RESERVE_AMOUNT

MAX_CANDIDATES_FOR_DIVERSIFICATION = 30
CORRELATION_PERIOD = "6mo"
HIGH_CORRELATION = 0.85
MEDIUM_CORRELATION = 0.70

SIGNAL_PRIORITY = {
    "POZYTYWNY": 3,
    "NEUTRALNY": 2,
    "NEGATYWNY": 1,
}

RISK_PRIORITY = {
    "UMIARKOWANE": 1,
    "PODWYZSZONE": 2,
    "WYSOKIE": 3,
}

RISK_FACTOR = {
    "UMIARKOWANE": 1.00,
    "PODWYZSZONE": 0.80,
    "WYSOKIE": 0.55,
}


def ranking_key(item):
    return (
        SIGNAL_PRIORITY.get(item.get("signal"), 0),
        item.get("score", 0),
        item.get("confirmed_days", 0),
        -RISK_PRIORITY.get(item.get("risk"), 99),
        item.get("change", 0),
    )


def money(value):
    return f"{value:,.2f} PLN".replace(",", " ").replace(".", ",")


def format_performance(statistics):
    lines = []

    for signal, horizons in statistics.items():
        lines.append(f"**{signal}**")

        for horizon, values in horizons.items():
            if values["count"] == 0:
                lines.append(f"{horizon}: za malo danych")
                continue

            median = values.get("median")

            median_text = (
                f" | mediana: {median:+.2f}%"
                if median is not None else ""
            )

            lines.append(
                f"{horizon}: srednio {values['average']:+.2f}%"
                f"{median_text} | dodatnie wyniki: "
                f"{values['positive']}% ({values['count']} prob)"
            )

    return "\n".join(lines)


def get_signal_summary(results):
    counts = {
        "POZYTYWNY": 0,
        "NEUTRALNY": 0,
        "NEGATYWNY": 0,
    }

    for item in results:
        signal = item.get("signal")

        if signal in counts:
            counts[signal] += 1

    return (
        f"Pozytywne: {counts['POZYTYWNY']}\n"
        f"Neutralne: {counts['NEUTRALNY']}\n"
        f"Negatywne: {counts['NEGATYWNY']}"
    )


# ==========================================
# WYBOR KANDYDATOW
# ==========================================

def get_candidate_pool(ranking):
    candidates = []

    for item in ranking:
        if item.get("signal") != "POZYTYWNY":
            continue

        if item.get("score", 0) < 7:
            continue

        if item.get("confirmed_days", 0) < 2:
            continue

        try:
            price = float(item["price"])

            if not math.isfinite(price) or price <= 0:
                continue

        except (KeyError, TypeError, ValueError):
            continue

        candidates.append(item)

    return candidates[:MAX_CANDIDATES_FOR_DIVERSIFICATION]


# ==========================================
# KORELACJA
# ==========================================

def get_return_series(ticker):
    try:
        history = yf.Ticker(ticker).history(
            period=CORRELATION_PERIOD,
            interval="1d",
            auto_adjust=True,
        )

        if history.empty or "Close" not in history.columns:
            return None

        close = pd.to_numeric(
            history["Close"],
            errors="coerce"
        ).dropna()

        close = close[close > 0]

        if len(close) < 30:
            return None

        returns = close.pct_change().dropna()

        if len(returns) < 25:
            return None

        index = pd.to_datetime(returns.index)

        if getattr(index, "tz", None) is not None:
            index = index.tz_localize(None)

        returns.index = index.normalize()

        return returns[
            ~returns.index.duplicated(keep="last")
        ]

    except Exception as error:
        print(
            f"KORELACJA {ticker}: "
            f"nie mozna pobrac danych: {error}"
        )
        return None


def build_correlation_matrix(candidates):
    series = {}

    for item in candidates:
        ticker = item["ticker"]
        values = get_return_series(ticker)

        if values is not None:
            series[ticker] = values

    if len(series) < 2:
        return {}, series

    frame = pd.concat(
        series,
        axis=1,
        join="inner"
    ).dropna()

    if len(frame) < 25:
        return {}, series

    matrix = frame.corr()

    correlations = {}
    tickers = list(matrix.columns)

    for i, first in enumerate(tickers):
        for second in tickers[i + 1:]:
            value = matrix.loc[first, second]

            if pd.notna(value):
                correlations[(first, second)] = float(value)
                correlations[(second, first)] = float(value)

    return correlations, series


def get_correlation(first, second, correlations):
    return correlations.get((first, second))


# ==========================================
# OCENA DYWERSYFIKACJI
# ==========================================

def candidate_utility(item, selected, correlations):
    score = float(item.get("score", 0))
    confirmation = float(item.get("confirmed_days", 0))
    risk = RISK_FACTOR.get(item.get("risk"), 0.55)

    utility = score + confirmation * 0.35 + risk

    sector = get_sector(item["ticker"])

    if sector != "NIEZWERYFIKOWANY":
        same_sector_count = sum(
            1
            for chosen in selected
            if get_sector(chosen["ticker"]) == sector
        )

        if same_sector_count:
            utility -= 1.5 * same_sector_count

    correlation_notes = []

    for chosen in selected:
        corr = get_correlation(
            item["ticker"],
            chosen["ticker"],
            correlations
        )

        if corr is None:
            continue

        correlation_notes.append((chosen["ticker"], corr))

        if corr >= HIGH_CORRELATION:
            utility -= 2.0

        elif corr >= MEDIUM_CORRELATION:
            utility -= 0.8

    return utility, correlation_notes


def select_diversified_finalists(ranking):
    candidates = get_candidate_pool(ranking)

    if not candidates:
        return [], {}, {}

    correlations, series = build_correlation_matrix(candidates)

    remaining = list(candidates)
    selected = []
    selection_details = {}

    while remaining and len(selected) < 3:
        evaluated = []

        for item in remaining:
            utility, corr_notes = candidate_utility(
                item,
                selected,
                correlations
            )

            evaluated.append((utility, item, corr_notes))

        evaluated.sort(
            key=lambda entry: (
                entry[0],
                ranking_key(entry[1]),
            ),
            reverse=True,
        )

        _, chosen, corr_notes = evaluated[0]

        selected.append(chosen)

        selection_details[chosen["ticker"]] = {
            "sector": get_sector(chosen["ticker"]),
            "correlations": corr_notes,
            "correlation_data_available": (
                chosen["ticker"] in series
            ),
        }

        remaining.remove(chosen)

    return selected, selection_details, correlations


# ==========================================
# WIRTUALNY PODZIAL KAPITALU
# ==========================================

def calculate_allocations(finalists):
    if not finalists:
        return []

    raw_weights = []

    for item in finalists:
        score_factor = item.get("score", 0) / 10

        confirmation_factor = (
            0.70 + 0.10 * item.get("confirmed_days", 0)
        )

        risk_factor = RISK_FACTOR.get(
            item.get("risk"),
            0.55
        )

        weight = (
            score_factor
            * confirmation_factor
            * risk_factor
        )

        raw_weights.append(max(weight, 0.01))

    total_weight = sum(raw_weights)

    allocations = []

    for item, weight in zip(finalists, raw_weights):
        amount = INVESTMENT_BUDGET * (weight / total_weight)

        price = float(item["price"])
        shares = math.floor(amount / price)
        cost = shares * price

        allocations.append({
            "ticker": item["ticker"],
            "price": price,
            "score": item.get("score", 0),
            "signal": item.get("signal"),
            "risk": item.get("risk"),
            "confirmed_days": item.get("confirmed_days", 0),
            "amount": amount,
            "shares": shares,
            "cost": cost,
            "unused": amount - cost,
        })

    return allocations


def calculate_portfolio_summary(allocations):
    total_cost = sum(
        item["cost"] for item in allocations
    )

    remaining = VIRTUAL_CAPITAL - total_cost

    if not allocations:
        risk = "BRAK POZYCJI"

    else:
        weighted_risk = sum(
            RISK_PRIORITY.get(item["risk"], 3)
            * item["cost"]
            for item in allocations
        )

        average_risk = (
            weighted_risk / total_cost
            if total_cost > 0 else 3
        )

        if average_risk <= 1.3:
            risk = "UMIARKOWANE"

        elif average_risk <= 2.3:
            risk = "PODWYZSZONE"

        else:
            risk = "WYSOKIE"

    return {
        "total_cost": total_cost,
        "remaining": remaining,
        "risk": risk,
    }


# ==========================================
# FORMATOWANIE RAPORTU
# ==========================================

def format_top10(ranking):
    if not ranking:
        return "Brak poprawnych analiz."

    lines = []

    for position, item in enumerate(
        ranking[:10],
        start=1
    ):
        lines.append(
            f"**{position}. {item['ticker']}** | "
            f"{item['score']}/10 | "
            f"{item['signal']} | "
            f"Ryzyko: {item['risk']} | "
            f"Potwierdzenie: "
            f"{item.get('confirmed_days', 0)}/3 | "
            f"{item['change']:+.2f}%"
        )

    return "\n".join(lines)


def format_one_finalist(item, allocation, details):
    sector = details.get(
        "sector",
        "NIEZWERYFIKOWANY"
    )

    lines = [
        f"**{item['ticker']}** | "
        f"{item['score']}/10 | "
        f"Ryzyko: {item['risk']} | "
        f"Potwierdzenie: "
        f"{item.get('confirmed_days', 0)}/3",

        f"Sektor: {sector}",

        f"Cena: {money(allocation['price'])} | "
        f"{allocation['shares']} szt. | "
        f"Wartosc: {money(allocation['cost'])}",
    ]

    strengths = item.get("strengths", [])

    if strengths:
        lines.append(
            "Mocne strony: " + "; ".join(strengths[:2])
        )

    risks = (
        item.get("risk_factors", [])
        + item.get("warnings", [])
    )

    risks = list(dict.fromkeys(risks))

    if risks:
        lines.append(
            "Ryzyka/uwagi: " + "; ".join(risks[:2])
        )

    correlations = details.get("correlations", [])

    if correlations:
        lines.append(
            "Korelacje: " + ", ".join(
                f"{ticker} {value:+.2f}"
                for ticker, value in correlations[:2]
            )
        )

    elif not details.get(
        "correlation_data_available",
        False
    ):
        lines.append(
            "Korelacja: brak wystarczajacych danych"
        )

    return "\n".join(lines)


def format_portfolio(allocations, summary):
    lines = [
        f"Kapital poczatkowy: {money(VIRTUAL_CAPITAL)}",
        f"Budzet maksymalny: {money(INVESTMENT_BUDGET)}",
        f"Szacowana wartosc pozycji: "
        f"{money(summary['total_cost'])}",
        f"Pozostala gotowka: "
        f"{money(summary['remaining'])}",
        f"Ocena ryzyka: {summary['risk']}",
    ]

    if not allocations:
        lines.append(
            "Nie zaproponowano zadnych pozycji."
        )

    return "\n".join(lines)


def format_diversification_note(
    finalists,
    selection_details
):
    if not finalists:
        return "Brak wybranych pozycji."

    sectors = [
        selection_details.get(
            item["ticker"],
            {}
        ).get(
            "sector",
            "NIEZWERYFIKOWANY"
        )
        for item in finalists
    ]

    known = [
        sector
        for sector in sectors
        if sector != "NIEZWERYFIKOWANY"
    ]

    unique_known = len(set(known))

    if len(known) < len(sectors):
        return (
            f"Rozpoznane sektory: {unique_known}. "
            "Czesc sektorow nie ma jeszcze "
            "zweryfikowanej klasyfikacji; "
            "dywersyfikacja sektorowa jest czesciowa."
        )

    return (
        f"Rozpoznane sektory: "
        f"{unique_known}/{len(sectors)} wybranych pozycji."
    )


# ==========================================
# DISCORD
# ==========================================

def send_discord_report(
    ranking,
    finalists,
    allocations,
    summary,
    results,
    failed,
    statistics,
    selection_details,
):
    webhook_url = os.getenv("DISCORD_WEBHOOK")

    if not webhook_url:
        print("\nDISCORD: Brak skonfigurowanego webhooka.")
        return

    fields = []

    if finalists and allocations:
        for position, (item, allocation) in enumerate(
            zip(finalists, allocations),
            start=1
        ):
            details = selection_details.get(
                item["ticker"],
                {}
            )

            fields.append({
                "name": f"TOP 3 - POZYCJA {position}",
                "value": format_one_finalist(
                    item,
                    allocation,
                    details
                )[:1024],
                "inline": False,
            })

    else:
        fields.append({
            "name": "FINALNA TOP 3",
            "value": (
                "Brak odpowiednich kandydatow. "
                "Kapital pozostaje wirtualna gotowka."
            ),
            "inline": False,
        })

    fields.extend([
        {
            "name": "WIRTUALNY PORTFEL",
            "value": format_portfolio(
                allocations,
                summary
            )[:1024],
            "inline": False,
        },
        {
            "name": "DYWERSYFIKACJA",
            "value": format_diversification_note(
                finalists,
                selection_details
            )[:1024],
            "inline": False,
        },
        {
            "name": "ROZKLAD SYGNALOW",
            "value": get_signal_summary(results),
            "inline": True,
        },
        {
            "name": "WYNIKI ANALIZ",
            "value": (
                f"Udane: {len(results)}\n"
                f"Nieudane: {len(failed)}\n"
                f"Instrumenty: {len(INSTRUMENTS)}"
            ),
            "inline": True,
        },
        {
            "name": "HISTORYCZNA SKUTECZNOSC",
            "value": (
                format_performance(statistics)[:1024]
                or "Brak danych"
            ),
            "inline": False,
        },
        {
            "name": "WAŻNE INFORMACJE",
            "value": (
                "Raport edukacyjny i symulacyjny SIGG. "
                "Podzial kapitalu jest wynikiem modelu, "
                "nie gwarancja. "
                "Korelacja historyczna nie przewiduje "
                "przyszlosci. Bot nie sklada zlecen."
            ),
            "inline": False,
        },
    ])

    embed = {
        "title": (
            f"{PROJECT_NAME} v{VERSION} "
            "- FINALNA ANALIZA SIGG"
        ),
        "description": (
            "**TOP 10 RYNKU**\n\n"
            + format_top10(ranking)
        )[:4000],
        "color": 5793266,
        "fields": fields,
        "footer": {
            "text": (
                f"MARKET AI {VERSION} | "
                "Analiza automatyczna | SIGG"
            )
        },
        "timestamp": datetime.now().astimezone().isoformat(),
    }

    payload = {
        "username": "MARKET AI",
        "embeds": [embed],
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": f"MARKET-AI/{VERSION}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=20
        ) as response:

            print(
                f"\nDISCORD: Status HTTP {response.status}"
            )

            if response.status in (200, 204):
                print(
                    "DISCORD: Raport wyslany pomyslnie."
                )

    except urllib.error.HTTPError as error:
        details = error.read().decode(
            "utf-8",
            errors="replace"
        )

        print(
            f"\nDISCORD: Blad HTTP: {error.code}"
        )
        print(f"DISCORD SZCZEGOLY: {details}")

    except urllib.error.URLError as error:
        print(
            f"\nDISCORD: Blad polaczenia: {error.reason}"
        )

    except Exception as error:
        print(
            f"\nDISCORD: Nieoczekiwany blad: {error}"
        )


# ==========================================
# GLOWNA FUNKCJA
# ==========================================

def start():
    print("=" * 65)
    print(f"{PROJECT_NAME} v{VERSION}")
    print(f"Data analizy: {datetime.now()}")
    print(f"Liczba instrumentow: {len(INSTRUMENTS)}")
    print("=" * 65)

    results = []
    failed = []

    print("\nROZPOCZYNAM ANALIZE RYNKU")

    for ticker in INSTRUMENTS:
        try:
            result = analyze_market(ticker)

            if result is not None:
                results.append(result)

            else:
                failed.append(ticker)

        except Exception as error:
            print(
                f"BLAD ANALIZY {ticker}: {error}"
            )
            failed.append(ticker)

    print("\n" + "=" * 75)
    print(f"RANKING {PROJECT_NAME} {VERSION}")
    print("=" * 75)

    ranking = sorted(
        results,
        key=ranking_key,
        reverse=True
    )

    for position, item in enumerate(
        ranking,
        start=1
    ):
        print(
            f"{position}. {item['ticker']} | "
            f"{item['signal']} | "
            f"{item['score']}/10 | "
            f"RYZYKO: {item['risk']} | "
            f"POTWIERDZENIE: "
            f"{item.get('confirmed_days', 0)}/3 | "
            f"{item['change']:+.2f}%"
        )

    print(
        "\nWYBOR FINALNEJ TOP 3 "
        "Z UWZGLEDNIENIEM DYWERSYFIKACJI"
    )

    finalists, selection_details, _ = (
        select_diversified_finalists(ranking)
    )

    allocations = calculate_allocations(finalists)

    summary = calculate_portfolio_summary(
        allocations
    )

    print(
        f"Wybrani kandydaci: {len(finalists)}"
    )

    for item in finalists:
        details = selection_details.get(
            item["ticker"],
            {}
        )

        print(
            f"- {item['ticker']} | "
            f"sektor: {details.get('sector')} | "
            f"ocena: {item['score']}/10 | "
            f"ryzyko: {item['risk']}"
        )

    print("\nPODSUMOWANIE")
    print(f"Udane analizy: {len(results)}")
    print(f"Nieudane analizy: {len(failed)}")

    print(
        f"Ocena ryzyka portfela: {summary['risk']}"
    )

    print(
        f"Pozostala gotowka: "
        f"{money(summary['remaining'])}"
    )

    if failed:
        print("\nINSTRUMENTY BEZ DANYCH:")

        for ticker in failed:
            print(f"- {ticker}")

    print("\nAKTUALIZACJA HISTORII")

    statistics = update_performance(results)

    send_discord_report(
        ranking,
        finalists,
        allocations,
        summary,
        results,
        failed,
        statistics,
        selection_details,
    )

    print("\nANALIZA ZAKONCZONA")
    print("=" * 75)


if __name__ == "__main__":
    start()
