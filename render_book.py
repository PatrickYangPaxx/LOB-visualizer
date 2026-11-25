"""Animate an MBP-10 book as a price ladder and save to MP4.

Overview, one video per file, one frame per --step seconds of market time:
    python render_book.py "D:/data/xnas-itch-20261026.mbp-10.dbn.zst"
    python render_book.py "D:/data/folder" --start 09:30 --end 16:00 --step 5

Tick-by-tick, one frame per finished book update, for a range spotted in the overview:
    python render_book.py "D:/data/folder" --date 20261028 --tick --start 10:47:00 --end 10:48:00
"""
import argparse
import glob
import math
import re
import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.dates import DateFormatter
import numpy as np
import pandas as pd

from lob_data import ASK_PX, ASK_SZ, BID_PX, BID_SZ, BOOK, TICK, TZ, load_seconds, load_ticks

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

BID_COLOR, ASK_COLOR = "#257de0", "#f47230"
TEXT, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#e6e6e6"
HALF_WINDOW = 0.14  # ladder shows mid +- this many dollars
MAX_MINUTES = 30


def find_files(args):
    # Windows shells don't expand globs, so accept a folder or a pattern too
    files = []
    for arg in args:
        p = Path(arg)
        if p.is_dir():
            files += p.glob("*.mbp-10.dbn.zst")
        elif any(ch in arg for ch in "*?"):
            files += map(Path, glob.glob(arg))
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"not found: {arg}")
    return sorted(files)


def day_of(path):
    m = re.search(r"(\d{8})", path.name)
    if not m:
        sys.exit(f"can't read a YYYYMMDD date from {path.name}")
    return m.group(1)


def at(day, hhmm):
    if hhmm.count(":") == 1:
        hhmm += ":00"
    return pd.Timestamp(day, tz=TZ) + pd.Timedelta(hhmm)


def fits(n, fps, force):
    minutes = n / fps / 60
    print(f"{n:,} frames -> {minutes:.1f} min video at {fps} fps")
    if minutes <= MAX_MINUTES or force:
        return True
    print(f"  over the {MAX_MINUTES}-minute cap: narrow the range, use --fps {math.ceil(n / MAX_MINUTES / 60)}, "
          f"or pass --force")
    return False


def seconds_for(path, cache_dir):
    cache = cache_dir / f"{path.name.split('.')[0]}.bars1s.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    print(f"decoding {path.name} (cached after the first run) ...")
    bars = load_seconds(path)
    bars.to_parquet(cache)
    return bars


def ticks_for(path, start, end, cache_dir):
    cache = cache_dir / f"{path.name.split('.')[0]}_{start:%H%M%S%f}-{end:%H%M%S%f}.ticks.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    print(f"decoding {path.name} {start:%H:%M:%S.%f} - {end:%H:%M:%S.%f} ...")
    ticks = load_ticks(path, start, end)
    ticks.to_parquet(cache)
    return ticks


def resample(bars, start, end, step):
    grid = pd.date_range(start, end, freq=f"{step}s", inclusive="left")
    # the book as it stood at each frame time; volume summed over the frame
    book = bars[BOOK].reindex(bars.index.union(grid)).ffill().loc[grid].dropna()
    vol = bars["volume"].resample(f"{step}s").sum().reindex(grid, fill_value=0)
    return book, vol.loc[book.index]


def render(book, x, info, clock, out, fps, title, change_px=None, change_side=None, trade_px=None):
    bid_px, ask_px = book[BID_PX].to_numpy(), book[ASK_PX].to_numpy()
    bid_sz, ask_sz = book[BID_SZ].to_numpy(), book[ASK_SZ].to_numpy()
    mid = (bid_px[:, 0] + ask_px[:, 0]) / 2
    # fixed x scale for the whole clip so bar lengths are comparable frame to frame
    xmax = np.percentile(np.concatenate([bid_sz, ask_sz]), 99.5) * 1.1

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.22,
                          left=0.07, right=0.97, top=0.86, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    ax_mid = fig.add_subplot(gs[1])

    bar_h = TICK * 0.8
    bids = ax.barh(bid_px[0], bid_sz[0], height=bar_h, color=BID_COLOR, edgecolor=TEXT, linewidth=0,
                   label="Bid (resting buy orders)")
    asks = ax.barh(ask_px[0], ask_sz[0], height=bar_h, color=ASK_COLOR, edgecolor=TEXT, linewidth=0,
                   label="Ask (resting sell orders)")
    mid_line = ax.axhline(mid[0], color=MUTED, lw=1, ls="--")
    trade_mark, = ax.plot([], [], ">", color=TEXT, ms=10)
    ax.set_xlim(0, xmax)
    ax.set_xlabel("Order size at level (shares)", color=MUTED)
    ax.set_ylabel("Price (USD)", color=MUTED)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    info_txt = ax.text(0.98, 0.97, "", transform=ax.transAxes, ha="right", va="top",
                       fontsize=10, color=TEXT, family="monospace")

    trail, = ax_mid.plot([], [], color=TEXT, lw=1.6)
    dot, = ax_mid.plot([], [], "o", color=TEXT, ms=8, mec="white", mew=2)
    ax_mid.set_xlim(x[0], x[-1] if len(x) > 1 else x[0] + 1)
    pad = (mid.max() - mid.min()) * 0.05 + TICK
    ax_mid.set_ylim(mid.min() - pad, mid.max() + pad)
    ax_mid.set_ylabel("Mid price (USD)", color=MUTED)
    if isinstance(x, pd.DatetimeIndex):
        ax_mid.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=x.tz))
    else:
        ax_mid.set_xlabel("Book update #", color=MUTED)
    ax_mid.grid(color=GRID, lw=0.8)
    ax_mid.set_title("Mid price so far", fontsize=10, color=MUTED, loc="left")

    for a in (ax, ax_mid):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=MUTED, labelsize=9)
        for s in ("left", "bottom"):
            a.spines[s].set_color(GRID)

    fig.suptitle(title, x=0.07, ha="left", fontsize=14, color=TEXT)
    clock_txt = fig.text(0.07, 0.905, "", fontsize=11, color=MUTED, family="monospace")

    writer = FFMpegWriter(fps=fps, bitrate=4000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "veryfast"])
    n = len(book)
    with writer.saving(fig, str(out), dpi=100):
        for i in range(n):
            for j, r in enumerate(bids):
                r.set_y(bid_px[i, j] - bar_h / 2)
                r.set_width(bid_sz[i, j])
            for j, r in enumerate(asks):
                r.set_y(ask_px[i, j] - bar_h / 2)
                r.set_width(ask_sz[i, j])
            if change_px is not None:
                # outline the level this update touched
                hit_bid = (change_side[i] == "B") & (np.abs(bid_px[i] - change_px[i]) < TICK / 2)
                hit_ask = (change_side[i] == "A") & (np.abs(ask_px[i] - change_px[i]) < TICK / 2)
                for r, hit in zip(bids, hit_bid):
                    r.set_linewidth(2 if hit else 0)
                for r, hit in zip(asks, hit_ask):
                    r.set_linewidth(2 if hit else 0)
            if trade_px is not None:
                trade_mark.set_data(([], []) if np.isnan(trade_px[i]) else ([xmax * 0.015], [trade_px[i]]))
            mid_line.set_ydata([mid[i], mid[i]])
            ax.set_ylim(mid[i] - HALF_WINDOW, mid[i] + HALF_WINDOW)

            info_txt.set_text(info[i])
            clock_txt.set_text(clock[i])
            trail.set_data(x[:i + 1], mid[:i + 1])
            dot.set_data([x[i]], [mid[i]])

            writer.grab_frame()
            if i % 500 == 0:
                print(f"  frame {i:,}/{n:,}")
    plt.close(fig)


def spread_line(book):
    ticks = np.round((book["ask_px_00"] - book["bid_px_00"]).to_numpy() / TICK).astype(int)
    return [f"spread {t} tick{'s' if t != 1 else ''}" for t in ticks]


def overview(path, args, out_dir):
    day = day_of(path)
    book, vol = resample(seconds_for(path, out_dir), at(day, args.start), at(day, args.end), args.step)
    if book.empty:
        print(f"{path.name}: no book between {args.start} and {args.end}")
        return
    if not fits(len(book), args.fps, args.force):
        return
    info = [f"{s}\ntraded {int(v):>7,} sh" for s, v in zip(spread_line(book), vol)]
    clock = [t.strftime("%Y-%m-%d  %H:%M:%S  New York") for t in book.index]
    span = f"{args.start.replace(':', '')}-{args.end.replace(':', '')}"
    mp4 = out_dir / f"intc_book_{day}_{span}_{args.step}s.mp4"
    render(book, book.index, info, clock, mp4, args.fps, "INTC limit order book, top 10 levels")
    print(f"saved {mp4}")


def tick_by_tick(path, args, out_dir):
    day = day_of(path)
    start, end = at(day, args.start), at(day, args.end)
    ticks = ticks_for(path, start, end, out_dir)
    if ticks.empty:
        sys.exit(f"{path.name}: no book updates between {args.start} and {args.end}")
    n = len(ticks)
    if not fits(n, args.fps, args.force):
        last_ok = ticks.index[MAX_MINUTES * 60 * args.fps - 1]
        sys.exit(f"  at {args.fps} fps, --end {last_ok:%H:%M:%S.%f} is the furthest that fits")

    info = [f"{s}\nupdate {i + 1:>7,}/{n:,}\nbuy  {int(b):>6,} sh\nsell {int(a):>6,} sh"
            for i, (s, b, a) in enumerate(zip(spread_line(ticks), ticks["buy_vol"], ticks["sell_vol"]))]
    clock = [t.strftime("%Y-%m-%d  %H:%M:%S.%f  New York") for t in ticks.index]
    span = f"{args.start.replace(':', '')}-{args.end.replace(':', '')}"
    mp4 = out_dir / f"intc_ticks_{day}_{span}.mp4"
    render(ticks, np.arange(1, n + 1), info, clock, mp4, args.fps, "INTC limit order book, tick by tick",
           change_px=ticks["change_px"].to_numpy(float), change_side=ticks["change_side"].to_numpy(),
           trade_px=ticks["trade_px"].to_numpy(float))
    print(f"saved {mp4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help=".mbp-10.dbn.zst files, a folder of them, or a glob pattern")
    ap.add_argument("--start", default="09:30", help="New York time HH:MM or HH:MM:SS[.ffffff]")
    ap.add_argument("--end", default="16:00")
    ap.add_argument("--step", type=int, default=5, help="overview: seconds of market time per frame")
    ap.add_argument("--tick", action="store_true", help="one frame per book update instead of per --step")
    ap.add_argument("--date", help="YYYYMMDD, picks one file out of several for --tick")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--force", action="store_true", help=f"render even if longer than {MAX_MINUTES} min")
    ap.add_argument("--out", default="lob_out", help="folder for the mp4s and cache")
    args = ap.parse_args()

    files = find_files(args.files)
    if args.date:
        files = [f for f in files if day_of(f) == args.date.replace("-", "")]
    if not files:
        ap.error("no matching files")
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    if args.tick:
        if len(files) != 1:
            ap.error(f"--tick renders one day but {len(files)} files match; pick one with --date YYYYMMDD")
        tick_by_tick(files[0], args, out_dir)
    else:
        for f in files:
            overview(f, args, out_dir)


if __name__ == "__main__":
    main()
