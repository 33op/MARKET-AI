import math
import warnings
from collections import Counter

import pandas as pd
import yfinance as yf


VERSION = "16.5"

INITIAL_CAPITAL = 20000.0
MAX_POSITIONS = 3
MAX_POSITION_WEIGHT = 0.30
MIN_CASH_WEIGHT = 0.10

HISTORY_PERIOD = "5y"
MIN_SESSIONS = 100
MIN_CALENDAR_SESSIONS = 150
MIN_SIGNAL_HISTORY = 60

COMMISSION_RATE = 0.003
SLIPPAGE_RATE = 0.001

HOLDING_PERIODS = [1, 3, 5]
DEFAULT_HOLDING_SESSIONS = 3

BENCHMARK = "^WIG"


# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================

def safe_float(value, default=0.0):
    try:
        result = float(value)
        if math.isnan(result) or math.isinf(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def normalize_index(index):
    index = pd.to_datetime(index)

    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)

    return index.normalize()


def download_history(ticker, period=HISTORY_PERIOD):
    """Pobiera skorygowane dane OHLCV potrzebne do symulacji."""
    try:
        data = yf.download(
            ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        required = ["Open", "Close", "High", "Low", "Volume"]

        if not all(column in data.columns for column in required):
            return None

        data = data[required].copy()
        data.index = normalize_index(data.index)
        data = data[~data.index.duplicated(keep="last")]
        data = data.sort_index()

        for column in required:
            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

        data = data.dropna(
        subset=["Open", "Close", "High", "Low", "Volume"]
)

        data = data[
       (data["Open"] > 0)
       & (data["Close"] > 0)
       & (data["High"] > 0)
       & (data["Low"] > 0)
       & (data["Volume"] >= 0)
