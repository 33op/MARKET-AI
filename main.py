
from datetime import datetime
from market_data import analyze_market

PROJECT_NAME = "MARKET AI"
VERSION = "1.0"

def start():
    print("=" * 45)
    print(f"{PROJECT_NAME} v{VERSION}")
    print(f"Data analizy: {datetime.now()}")
    print("=" * 45)

    print("\nROZPOCZYNAM ANALIZE RYNKU")

    tickers = ["PKO.WA", "PZU.WA", "CDR.WA", "KGH.WA", "ALE.WA"]

    for ticker in tickers:
        analyze_market(ticker)

    print("\nANALIZA ZAKONCZONA")
    print("=" * 45)


if __name__ == "__main__":
    start()
