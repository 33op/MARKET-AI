
import os
import json
import math
import urllib.request
import urllib.error
from datetime import datetime

from market_data import analyze_market
from instruments import INSTRUMENTS
from performance import update_performance

PROJECT_NAME = "MARKET AI"
VERSION = "11.0"

VIRTUAL_CAPITAL = 20000.00
RESERVE_AMOUNT = 2000.00
INVESTMENT_BUDGET = VIRTUAL_CAPITAL - RESERVE_AMOUNT

SIGNAL_PRIORITY = {
    "POZYTYWNY": 3,
    "NEUTRALNY": 2,
    "NEGATYWNY": 1
}

RISK_PRIORITY = {
    "UMIARKOWANE": 1,
    "PODWYZSZONE": 2,
    "WYSOKIE": 3
}

RISK_FACTOR = {
    "UMIARKOWANE": 1.00,
    "PODWYZSZONE": 0.80,
    "WYSOKIE": 0.55
}


def ranking_key(item):
    return (
        SIGNAL_PRIORITY.get(item.get("signal"), 0),
        item.get("score", 0),
        item.get("confirmed_days", 0),
        -RISK_PRIORITY.get(item.get("risk"), 99),
        item.get("change", 0)
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
                if median is not None
                else ""
            )

            lines.append(
                f"{horizon}: srednio "
                f"{values['average']:+.2f}%"
                f"{median_text} | "
                f"dodatnie wyniki: {values['positive']}% "
                f"({values['count']} prob)"
            )

    return "\n".join(lines)


def get_signal_summary(results):
    counts = {
        "POZYTYWNY": 0,
        "NEUTRALNY": 0,
        "NEGATYWNY": 0
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


def select_finalists(ranking):
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

    return candidates[:3]


def calculate_allocations(finalists):
    if not finalists:
        return []

    # Wagi uwzgledniaja punktacje, potwierdzenie i ryzyko.
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
        share = weight / total_weight
        amount = INVESTMENT_BUDGET * share

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
            "unused": amount - cost
        })

    return allocations


def calculate_portfolio_summary(allocations):
    total_cost = sum(item["cost"] for item in allocations)
    remaining = VIRTUAL_CAPITAL - total_cost

    if not allocations:
        risk = "BRAK POZYCJI"
    else:
        risk_levels = [
            RISK_PRIORITY.get(item["risk"], 3)
            for item in allocations
        ]

        average_risk = sum(risk_levels) / len(risk_levels)

        if average_risk <= 1.3:
            risk = "UMIARKOWANE"
        elif average_risk <= 2.3:
            risk = "PODWYZSZONE"
        else:
            risk = "WYSOKIE"

    return {
        "total_cost": total_cost,
        "remaining": remaining,
        "risk": risk
    }


def format_top10(ranking):
    if not ranking:
        return "Brak poprawnych analiz."

    lines = []

    for position, item in enumerate(ranking[:10], start=1):
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


def format_finalists(finalists, allocations):
    if not allocations:
        return (
            "Brak trzech odpowiednich kandydatow. "
            "Kapital pozostaje wirtualna gotowka."
        )

    lines = []

    for position, (item, allocation) in enumerate(
        zip(finalists, allocations),
        start=1
    ):
        lines.append(
            f"**{position}. {item['ticker']}**\n"
            f"Ocena: {item['score']}/10 | "
            f"Ryzyko: {item['risk']}\n"
            f"Potwierdzenie: "
            f"{item.get('confirmed_days', 0)}/3\n"
            f"Cena: {money(allocation['price'])}\n"
            f"Wirtualny zakup: {allocation['shares']} szt.\n"
            f"Wartosc: {money(allocation['cost'])}"
        )

        warnings = item.get("warnings", [])

        if warnings:
            lines.append(
                "Uwagi: " + "; ".join(warnings[:3])
            )

    return "\n\n".join(lines)


def format_portfolio(allocations, summary):
    lines = [
        f"Kapital poczatkowy: {money(VIRTUAL_CAPITAL)}",
        f"Przydzielony budzet: {money(INVESTMENT_BUDGET)}",
        f"Szacowana wartosc zakupow: "
        f"{money(summary['total_cost'])}",
        f"Pozostala gotowka: {money(summary['remaining'])}",
        f"Ocena ryzyka: {summary['risk']}"
    ]

    if not allocations:
        lines.append("Nie zaproponowano zadnych pozycji.")

    return "\n".join(lines)


def send_discord_report(
    ranking,
    finalists,
    allocations,
    summary,
    results,
    failed,
    statistics
):
    webhook_url = os.getenv("DISCORD_WEBHOOK")

    if not webhook_url:
        print("\nDISCORD: Brak skonfigurowanego webhooka.")
        return

    embed = {
        "title": f"{PROJECT_NAME} v{VERSION} - FINALNA ANALIZA SIGG",
        "description": (
            "**TOP 10 RYNKU**\n\n"
            + format_top10(ranking)
        )[:4000],
        "color": 5793266,
        "fields": [
            {
                "name": "FINALNA TOP 3",
                "value": format_finalists(
                    finalists,
                    allocations
                )[:1024],
                "inline": False
            },
            {
                "name": "WIRTUALNY PORTFEL",
                "value": format_portfolio(
                    allocations,
                    summary
                )[:1024],
                "inline": False
            },
            {
                "name": "ROZKLAD SYGNALOW",
                "value": get_signal_summary(results),
                "inline": True
            },
            {
                "name": "WYNIKI ANALIZ",
                "value": (
                    f"Udane: {len(results)}\n"
                    f"Nieudane: {len(failed)}\n"
                    f"Instrumenty: {len(INSTRUMENTS)}"
                ),
                "inline": True
            },
            {
                "name": "HISTORYCZNA SKUTECZNOSC",
                "value": format_performance(statistics)[:1024]
                or "Brak danych",
                "inline": False
            },
            {
                "name": "WAŻNE INFORMACJE",
                "value": (
                    "Raport edukacyjny i symulacyjny SIGG. "
                    "Podzial kapitalu jest propozycja modelu, "
                    "nie gwarancja wyniku. "
                    "Bot nie sklada zlecen."
                ),
                "inline": False
            }
        ],
        "footer": {
            "text": (
                "MARKET AI 11.0 | "
                "Analiza automatyczna | SIGG"
            )
        },
        "timestamp": datetime.now().astimezone().isoformat()
    }

    payload = {
        "username": "MARKET AI",
        "embeds": [embed]
    }

    data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "MARKET-AI/11.0"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            print(f"\nDISCORD: Status HTTP {response.status}")

            if response.status in (200, 204):
                print("DISCORD: Raport wyslany pomyslnie.")

    except urllib.error.HTTPError as error:
        details = error.read().decode(
            "utf-8",
            errors="replace"
        )

        print(f"\nDISCORD: Blad HTTP: {error.code}")
        print(f"DISCORD SZCZEGOLY: {details}")

    except urllib.error.URLError as error:
        print(f"\nDISCORD: Blad polaczenia: {error.reason}")

    except Exception as error:
        print(f"\nDISCORD: Nieoczekiwany blad: {error}")


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
            print(f"BLAD ANALIZY {ticker}: {error}")
            failed.append(ticker)

    print("\n" + "=" * 75)
    print(f"RANKING {PROJECT_NAME} {VERSION}")
    print("=" * 75)

    ranking = sorted(
        results,
        key=ranking_key,
        reverse=True
    )

    for position, item in enumerate(ranking, start=1):
        print(
            f"{position}. {item['ticker']} | "
            f"{item['signal']} | "
            f"{item['score']}/10 | "
            f"RYZYKO: {item['risk']} | "
            f"POTWIERDZENIE: "
            f"{item.get('confirmed_days', 0)}/3 | "
            f"{item['change']:+.2f}%"
        )

    print("\nWYBIERANIE FINALNEJ TOP 3")

    finalists = select_finalists(ranking)
    allocations = calculate_allocations(finalists)
    summary = calculate_portfolio_summary(allocations)

    print(f"Wybrani kandydaci: {len(finalists)}")

    for item in allocations:
        print(
            f"{item['ticker']}: "
            f"{item['shares']} szt. | "
            f"{money(item['cost'])} | "
            f"Ryzyko: {item['risk']}"
        )

    print("\nPODSUMOWANIE")
    print(f"Udane analizy: {len(results)}")
    print(f"Nieudane analizy: {len(failed)}")
    print(f"Ocena ryzyka portfela: {summary['risk']}")
    print(f"Pozostala gotowka: {money(summary['remaining'])}")

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
        statistics
    )

    print("\nANALIZA ZAKONCZONA")
    print("=" * 75)


if __name__ == "__main__":
    start()
