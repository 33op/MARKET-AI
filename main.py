
from datetime import datetime
from market_data import analyze_market

PROJECT_NAME = "MARKET AI"
VERSION = "4.0"


def start():
    print("=" * 50)
    print(f"{PROJECT_NAME} v{VERSION}")
    print(f"Data analizy: {datetime.now()}")
    print("=" * 50)

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
    print("=" * 65)
    print("RANKING MARKET AI 4.0")
    print("=" * 65)

    ranking = sorted(
        results,
        key=lambda x: (
            x["score"],
            x["change"]
        ),
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

    print("\nPODSUMOWANIE")

    positive = sum(
        1 for item in results
        if item["signal"] == "POZYTYWNY"
    )

    negative = sum(
        1 for item in results
        if item["signal"] == "NEGATYWNY"
    )

    print(f"Przeanalizowane spółki: {len(results)}")
    print(f"Pozytywne sygnały: {positive}")
    print(f"Negatywne sygnały: {negative}")

    print("\nANALIZA ZAKONCZONA")
    print("=" * 65)


if __name__ == "__main__":
    start()
