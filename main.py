
from datetime import datetime

PROJECT_NAME = "MARKET AI"
VERSION = "1.0"

def start():
    print("=" * 40)
    print(f"{PROJECT_NAME} v{VERSION}")
    print("=" * 40)

    print("System uruchomiony.")
    print(f"Data: {datetime.now()}")
    print("Status: ONLINE")
    print("Moduły analizy: w przygotowaniu")
    print("Źródła danych: w przygotowaniu")
    print("Discord: w przygotowaniu")

    print("=" * 40)
    print("MARKET AI is ready!")

if __name__ == "__main__":
    start()
