"""Decode Databento MBP-10 files into book snapshots."""
import pandas as pd

TZ = "America/New_York"
TICK = 0.01
LEVELS = range(10)
BID_PX = [f"bid_px_{i:02d}" for i in LEVELS]
ASK_PX = [f"ask_px_{i:02d}" for i in LEVELS]
BID_SZ = [f"bid_sz_{i:02d}" for i in LEVELS]
ASK_SZ = [f"ask_sz_{i:02d}" for i in LEVELS]
BOOK = BID_PX + ASK_PX + BID_SZ + ASK_SZ
# Records sharing a timestamp are one event; only the one flagged F_LAST shows the finished book.
F_LAST = 128


def chunks(path, size=2_000_000):
    import databento as db

    store = db.DBNStore.from_file(path)
    for df in store.to_df(price_type="float", pretty_ts=True, map_symbols=False, tz=TZ, count=size):
        yield df[["action", "side", "price", "size", "flags"] + BOOK]


def valid(df):
    return df[(df["bid_px_00"] > 0) & (df["ask_px_00"] > df["bid_px_00"])]


def load_seconds(path):
    """1s bars: the book as it stood at the end of each second, plus shares traded during it."""
    parts = []
    for df in chunks(path):
        df = valid(df)
        if df.empty:
            continue
        sec = df.index.floor("s")
        part = df[BOOK].groupby(sec).last()
        part["volume"] = df["size"].where(df["action"] == "T", 0).groupby(sec).sum()
        parts.append(part)
    bars = pd.concat(parts)
    # a second split across two chunks shows up twice: keep the last book, sum the volume
    volume = bars["volume"].groupby(level=0).sum()
    return bars[BOOK].groupby(level=0).last().join(volume)


def load_ticks(path, start, end):
    """One row per finished book update in [start, end), with the trades that happened in it."""
    parts = []
    for df in chunks(path):
        if df.index[-1] < start:
            continue
        parts.append(df[(df.index >= start) & (df.index < end)])
        if df.index[-1] >= end:
            break
    df = valid(pd.concat(parts)) if parts else pd.DataFrame()
    if df.empty:
        return df

    is_last = (df["flags"] & F_LAST) > 0
    event = is_last.shift(fill_value=False).cumsum().to_numpy()
    trade = df["action"] == "T"
    # Databento trade side is the aggressor: B = buyer lifted the ask
    buy = df["size"].where(trade & (df["side"] == "B"), 0)
    sell = df["size"].where(trade & (df["side"] == "A"), 0)
    change = df["action"].isin(["A", "C", "M"])

    ticks = df.loc[~pd.Series(event).duplicated(keep="last").to_numpy(), BOOK].copy()
    ticks["buy_vol"] = buy.groupby(event).sum().to_numpy()
    ticks["sell_vol"] = sell.groupby(event).sum().to_numpy()
    ticks["trade_px"] = df["price"].where(trade).groupby(event).last().to_numpy()
    ticks["change_px"] = df["price"].where(change).groupby(event).last().to_numpy()
    ticks["change_side"] = df["side"].where(change).groupby(event).last().to_numpy()
    return ticks
