
import os
import json
import urllib.request
import urllib.error
from datetime import datetime

from market_data import analyze_market
from instruments import INSTRUMENTS
from performance import update_performance

PROJECT_NAME = "MARKET AI"
VERSION = "10.0"


# Im wyższy priorytet, tym wyżej instrument w rankingu.
SIGNAL_PRIORITY = {
    "POZYTYWNY": 3,
    "NEUTRALNY": 2,
    "NEGATYWNY": 1
}

# Niższe ryzyko ma pierwszeństwo.
RISK_PRIORITY = {
    "UMIARKOWANE": 1,
    "PODWYZSZONE": 2,
    "WYSOKIE": 3
}


def ranking_key(item):
    signal = item.get("signal", "BRAK")
    risk = item.get("risk", "NIEZNANE")

    signal_priority = SIGNAL_PRIORITY.get(signal, 0)
    risk_priority = RISK_PRIORITY.get(risk, 99)

    return (
        signal_priority,
        item.get("score", 0),
        item.get("confirmed_days", 0),
        -risk_priority,
        item.get("change", 0)
    )


def format_performance(statistics):
    lines = []

    for signal, horizons in statistics.items():
        lines.append(f"**{signal}**")

        for horizon, values in horizons.items():
            if values["count"] == 0:
                lines.append(f"{horizon}: za malo danych")
            else:
                lines.append(
                    f"{horizon}: srednio "
                    f"{values['average']:+.2f}% | "
                    f"dodatnie wyniki: {values['positive']}% "
                    f"({values['count']} prob)"
                )

    return "\n".join(lines)


def format_signal_summary(results):
    counts = {
        "POZYTYWNY": 0,
        "NEUTRALNY": 0,
        "NEGATYWNY": 0
    }

    for item in results:
        signal = item.get("signal", "BRAK")

        if signal in counts:
            counts[signal] += 1

    return (
        f"Pozytywne: {counts['POZYTYWNY']}\n"
        f"Neutralne: {counts['NEUTRALNY']}\n"
        f"Negatywne: {counts['NEGATYWNY']}"
    )


def send_discord_report(ranking, results, failed, statistics):
    webhook_url = os.getenv("DISCORD_WEBHOOK")

    if not webhook_url:
        print("\nDISCORD: Brak skonfigurowanego webhooka.")
        return

    top_10 = ranking[:10]

    if top_10:
        description = "\n".join(
            f"**{i}. {item['ticker']}** | "
            f"{item['score']}/10 | "
            f"{item['signal']} | "
            f"Ryzyko: {item['risk']} | "
            f"Potwierdzenie: {item.get('confirmed_days', 0)}/3 | "
            f"{item['change']:+.2f}%"
            for i, item in enumerate(top_10, start=1)
        )
    else:
        description = "Nie udało się przeanalizować żadnych instrumentów."

    performance_text = format_performance(statistics)

    embed = {
        "title": f"{PROJECT_NAME} v{VERSION} - RAPORT",
        "description": description,
        "color": 5793266,
        "fields": [
            {
                "name": "Udane analizy",
                "value": str(len(results)),
                "inline": True
            },
            {
                "name": "Nieudane analizy",
                "value": str(len(failed)),
                "inline": True
            },
            {
                "name": "Przeanalizowane instrumenty",
                "value": str(len(INSTRUMENTS)),
                "inline": True
            },
            {
                "name": "Rozkład sygnałów",
                "value": format_signal_summary(results),
                "inline": True
            },
            {
                "name": "Skuteczność wcześniejszych analiz",
                "value": performance_text[:1024] or "Brak danych",
                "inline": False
            }
        ],
        "footer": {
            "text": "MARKET AI 10.0 | Ranking wedlug priorytetu sygnalu"
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
            "User-Agent": "MARKET-AI/10.0"
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            print(f"\nDISCORD: Status HTTP {response.status}")

            if response.status in (200, 204):
                print("DISCORD: Raport wysłany pomyślnie.")

    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        print(f"\nDISCORD: Błąd HTTP: {error.code}")
        print(f"DISCORD SZCZEGÓŁY: {details}")

    except urllib.error.URLError as error:
        print(f"\nDISCORD: Błąd połączenia: {error.reason}")

    except Exception as error:
        print(f"\nDISCORD: Nieoczekiwany błąd: {error}")


def start():
    print("=" * 60)
    print(f"{PROJECT_NAME} v{VERSION}")
    print(f"Data analizy: {datetime.now()}")
    print(f"Liczba instrumentow: {len(INSTRUMENTS)}")
    print("=" * 60)

    results = []
    failed = []

    print("\nROZPOCZYNAM ANALIZE RYNKU")

    for ticker in INSTRUMENTS:
        result = analyze_market(ticker)

        if result is not None:
            results.append(result)
        else:
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
            f"POTWIERDZENIE: {item.get('confirmed_days', 0)}/3 | "
            f"{item['change']:+.2f}%"
        )

    print("\nPODSUMOWANIE")
    print(f"Udane analizy: {len(results)}")
    print(f"Nieudane analizy: {len(failed)}")
    print("\nROZKLAD SYGNALOW")
    print(format_signal_summary(results))

    if failed:
        print("\nINSTRUMENTY BEZ DANYCH:")
        for ticker in failed:
            print(f"- {ticker}")

    print("\nAKTUALIZACJA HISTORII I TESTOWANIE ANALIZ")
    statistics = update_performance(results)

    send_discord_report(
        ranking,
        results,
        failed,
        statistics
    )

    print("\nANALIZA ZAKONCZONA")
    print("=" * 75)


if __name__ == "__main__":
    start()
