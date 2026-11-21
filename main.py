from datetime import datetime
from market_data import analyze_market

PROJECT_NAME = "MARKET AI"
VERSION = "2.0"


def start():
    print("=" * 45)
    print(f"{PROJECT_NAME} v{VERSION}")
    print(f"Data analizy: {datetime.now()}")
    print("=" * 45)

    print("\nROZPOCZYNAM ANALIZE RYNKU")

    tickers = ["PKO.WA", "PZU.WA", "CDR.WA", "KGH.WA", "ALE.WA"]

    results = []

    for ticker in tickers:
        result = analyze_market(ticker)

        if result is not None:
            results.append(result)

    print("\n")
    print("=" * 45)
    print("RANKING MARKET AI")
    print("=" * 45)

    ranking = sorted(
        results,
        key=lambda x: (x["score"], x["change"]),
        reverse=True
    )

    for position, item in enumerate(ranking, start=1):
        print(
            f"{position}. {item['ticker']} | "
            f"{item['signal']} | "
            f"{item['score']}/3 | "
            f"{item['change']:+.2f}%"
        )

    print("\nANALIZA ZAKONCZONA")
    print("=" * 45)


if __name__ == "__main__":
    start()
