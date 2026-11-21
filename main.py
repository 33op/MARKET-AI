
from datetime import datetime
from market_data import analyze_market

PROJECT_NAME = "MARKET AI"
VERSION = "5.0"


def start():
    print("=" * 55)
    print(f"{PROJECT_NAME} v{VERSION}")
    print(f"Data analizy: {datetime.now()}")
    print("=" * 55)

    tickers = [
        "PKO.WA",
        "PZU.WA",
        "CDR.WA",
        "KGH.WA",
        "ALE.WA"
    ]

    results = []

    print("\nROZPOCZYNAM ANALIZE RYNKU")

    for ticker in tickers:
        result = analyze_market(ticker)

        if result is not None:
            results.append(result)

    print("\n")
    print("=" * 70)
    print("RANKING MARKET AI 5.0")
    print("=" * 70)

    ranking = sorted(
        results,
        key=lambda item: (item["score"], item["change"]),
        reverse=True
    )

    for position, item in enumerate(ranking, start=1):
        print(
            f"{position}. {item['ticker']} | "
            f"{item['signal']} | "
            f"{item['score']}/10 | "
            f"RYZYKO: {item['risk']} | "
            f"{item['change']:+.2f}%"
        )

    positive = sum(
        item["signal"] == "POZYTYWNY"
        for item in results
    )

    negative = sum(
        item["signal"] == "NEGATYWNY"
        for item in results
    )

    print("\nPODSUMOWANIE")
    print(f"Przeanalizowane spolki: {len(results)}")
    print(f"Pozytywne sygnaly: {positive}")
    print(f"Negatywne sygnaly: {negative}")

    print("\nANALIZA ZAKONCZONA")
    print("=" * 70)


if __name__ == "__main__":
    start()
