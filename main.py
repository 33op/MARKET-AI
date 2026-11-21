
from datetime import datetime
from market_data import analyze_market
from instruments import INSTRUMENTS

PROJECT_NAME = "MARKET AI"
VERSION = "7.0"


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

    print("\n")
    print("=" * 75)
    print("RANKING MARKET AI 6.0")
    print("=" * 75)

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

    print("\nPODSUMOWANIE")
    print(f"Udane analizy: {len(results)}")
    print(f"Nieudane analizy: {len(failed)}")

    if failed:
        print("\nINSTRUMENTY BEZ DANYCH:")
        for ticker in failed:
            print(f"- {ticker}")

    print("\nANALIZA ZAKONCZONA")
    print("=" * 75)


if __name__ == "__main__":
    start()
