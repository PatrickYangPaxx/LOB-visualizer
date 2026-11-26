"""Scrub through a day's book, mark a range, and render it tick by tick.

    python viewer.py "D:/data/folder"

Click the mid-price chart or drag the slider to move. Zoom the chart with the toolbar's magnifier.
Keys: left/right = 1 s, shift+left/right = 1 min, s = set start, e = set end, enter = render.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import pandas as pd

from lob_data import ASK_PX, ASK_SZ, BID_PX, BID_SZ, TICK
from render_book import (ASK_COLOR, BID_COLOR, GRID, HALF_WINDOW, MAX_MINUTES, MUTED, TEXT,
                         at, day_of, find_files, resample, seconds_for)

# free up the keys we use from matplotlib's default shortcuts
plt.rcParams["keymap.back"] = ["backspace"]
plt.rcParams["keymap.forward"] = []
plt.rcParams["keymap.save"] = ["ctrl+s"]
plt.rcParams["keymap.home"] = ["h", "home"]

SELECT_COLOR = "#f2a600"
WARN_COLOR = "#ec4647"
ONE_SEC = pd.Timedelta("1s")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


class Viewer:
    def __init__(self, files, out_dir, fps):
        self.files, self.out_dir, self.fps = files, out_dir, fps
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title("LOB viewer")
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.15,
                                   left=0.06, right=0.98, top=0.86, bottom=0.25)
        self.ax_book = self.fig.add_subplot(gs[0])
        self.ax_day = self.fig.add_subplot(gs[1])

        self.slider = Slider(self.fig.add_axes([0.06, 0.15, 0.92, 0.03]), "", 0, 1, valinit=0, valstep=1,
                             color=MUTED, initcolor="none")
        self.slider.valtext.set_visible(False)
        self.slider.on_changed(lambda v: self.move(int(v)))

        actions = [("< day", lambda _: self.load(self.day_i - 1)),
                   ("day >", lambda _: self.load(self.day_i + 1)),
                   ("Set start [s]", lambda _: self.mark("start")),
                   ("Set end [e]", lambda _: self.mark("end")),
                   ("Render ticks [enter]", lambda _: self.render())]
        self.buttons = []  # keep references: widgets stop responding once garbage collected
        for k, (label, fn) in enumerate(actions):
            b = Button(self.fig.add_axes([0.06 + k * 0.115, 0.03, 0.105, 0.055]), label)
            b.on_clicked(fn)
            self.buttons.append(b)
        self.status = self.fig.text(0.06, 0.11, "", va="center", fontsize=10, color=TEXT)

        self.title = self.fig.text(0.06, 0.95, "", fontsize=14, color=TEXT)
        self.clock = self.fig.text(0.06, 0.905, "", fontsize=11, color=MUTED, family="monospace")
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.load(0)

    def load(self, day_i):
        if not 0 <= day_i < len(self.files):
            return
        self.day_i = day_i
        self.path = self.files[day_i]
        day = day_of(self.path)
        bars = seconds_for(self.path, self.out_dir)
        book, _ = resample(bars, at(day, "09:30"), at(day, "16:00"), 1)
        self.times = book.index
        self.updates = bars["updates"].reindex(self.times, fill_value=0).to_numpy()
        self.bid_px, self.ask_px = book[BID_PX].to_numpy(), book[ASK_PX].to_numpy()
        self.bid_sz, self.ask_sz = book[BID_SZ].to_numpy(), book[ASK_SZ].to_numpy()
        self.mid = (self.bid_px[:, 0] + self.ask_px[:, 0]) / 2
        self.start = self.end = None
        self.marks = []

        ax = self.ax_book
        ax.cla()
        xmax = np.percentile(np.concatenate([self.bid_sz, self.ask_sz]), 99.5) * 1.1
        self.bar_h = TICK * 0.8
        self.bids = ax.barh(self.bid_px[0], self.bid_sz[0], height=self.bar_h, color=BID_COLOR, label="Bid")
        self.asks = ax.barh(self.ask_px[0], self.ask_sz[0], height=self.bar_h, color=ASK_COLOR, label="Ask")
        self.mid_line = ax.axhline(self.mid[0], color=MUTED, lw=1, ls="--")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("Order size at level (shares)", color=MUTED)
        ax.set_ylabel("Price (USD)", color=MUTED)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
        ax.legend(loc="lower right", frameon=False, fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.8)
        self.info = ax.text(0.98, 0.97, "", transform=ax.transAxes, ha="right", va="top",
                            fontsize=10, color=TEXT, family="monospace")
        style(ax)

        ax = self.ax_day
        ax.cla()
        ax.plot(self.times, self.mid, color=TEXT, lw=0.8)
        self.cursor = ax.axvline(self.times[0], color=MUTED, lw=1)
        ax.set_xlim(self.times[0], self.times[-1])
        locator = mdates.AutoDateLocator(tz=self.times.tz)
        ax.xaxis.set_major_locator(locator)
        # adds seconds to the labels as you zoom in
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, tz=self.times.tz, show_offset=False))
        ax.set_ylabel("Mid price (USD)", color=MUTED)
        ax.set_title("Click to jump. Zoom with the toolbar magnifier.", fontsize=10, color=MUTED, loc="left")
        ax.grid(color=GRID, lw=0.8)
        style(ax)

        self.title.set_text(f"INTC limit order book  |  day {day_i + 1}/{len(self.files)}  |  {self.path.name}")
        self.slider.valmax = len(self.times) - 1
        self.slider.ax.set_xlim(0, self.slider.valmax)
        self.set_status("Mark a start and an end, then render.")
        self.slider.set_val(0)

    def move(self, i):
        self.i = i
        for j, r in enumerate(self.bids):
            r.set_y(self.bid_px[i, j] - self.bar_h / 2)
            r.set_width(self.bid_sz[i, j])
        for j, r in enumerate(self.asks):
            r.set_y(self.ask_px[i, j] - self.bar_h / 2)
            r.set_width(self.ask_sz[i, j])
        self.mid_line.set_ydata([self.mid[i], self.mid[i]])
        self.ax_book.set_ylim(self.mid[i] - HALF_WINDOW, self.mid[i] + HALF_WINDOW)
        self.cursor.set_xdata([self.times[i], self.times[i]])

        spread = round((self.ask_px[i, 0] - self.bid_px[i, 0]) / TICK)
        self.info.set_text(f"spread {spread} tick{'s' if spread != 1 else ''}\n"
                           f"updates this second {self.updates[i]:>5,}")
        self.clock.set_text(self.times[i].strftime("%Y-%m-%d  %H:%M:%S  New York"))
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        toolbar_busy = getattr(self.fig.canvas.toolbar, "mode", "")
        if event.inaxes is not self.ax_day or event.button != 1 or toolbar_busy:
            return
        t = pd.Timestamp(mdates.num2date(event.xdata))
        self.slider.set_val(min(self.times.searchsorted(t), len(self.times) - 1))

    def on_key(self, event):
        steps = {"left": -1, "right": 1, "shift+left": -60, "shift+right": 60}
        if event.key in steps:
            self.slider.set_val(int(np.clip(self.i + steps[event.key], 0, len(self.times) - 1)))
        elif event.key == "s":
            self.mark("start")
        elif event.key == "e":
            self.mark("end")
        elif event.key == "enter":
            self.render()

    def mark(self, which):
        setattr(self, which, self.i)
        if self.start is not None and self.end is not None and self.start > self.end:
            self.start, self.end = self.end, self.start
        for m in self.marks:
            m.remove()
        a = self.start if self.start is not None else self.end
        b = self.end if self.end is not None else self.start
        # the end second is included, so the range runs to the end of it
        lo, hi = self.times[a], self.times[b] + ONE_SEC
        # lines as well as the band: a short range is under a pixel wide when the whole day is shown
        self.marks = [self.ax_day.axvspan(lo, hi, color=SELECT_COLOR, alpha=0.35, lw=0),
                      self.ax_day.axvline(lo, color=SELECT_COLOR, lw=1.5),
                      self.ax_day.axvline(hi, color=SELECT_COLOR, lw=1.5)]

        if self.start is None or self.end is None:
            self.set_status(f"{which} {self.times[self.i]:%H:%M:%S}, now mark the {'end' if which == 'start' else 'start'}")
        else:
            n, minutes = self.estimate()
            msg = (f"{self.times[self.start]:%H:%M:%S} - {(self.times[self.end] + ONE_SEC):%H:%M:%S}  |  "
                   f"{n:,} updates -> {minutes:.1f} min at {self.fps} fps")
            self.set_status(msg + ("  (over the cap)" if minutes > MAX_MINUTES else ""), warn=minutes > MAX_MINUTES)
        self.fig.canvas.draw_idle()

    def estimate(self):
        n = int(self.updates[self.start:self.end + 1].sum())
        return n, n / self.fps / 60

    def render(self):
        if self.start is None or self.end is None:
            self.set_status("Mark a start and an end first.", warn=True)
            return
        n, minutes = self.estimate()
        if minutes > MAX_MINUTES:
            self.set_status(f"{minutes:.0f} min is over the {MAX_MINUTES}-minute cap. Narrow the range.", warn=True)
            return
        start = self.times[self.start].strftime("%H:%M:%S")
        end = (self.times[self.end] + ONE_SEC).strftime("%H:%M:%S")
        cmd = [sys.executable, str(Path(__file__).with_name("render_book.py")), str(self.path), "--tick",
               "--start", start, "--end", end, "--fps", str(self.fps), "--out", str(self.out_dir)]
        # separate process so the window stays usable; progress prints in the terminal
        subprocess.Popen(cmd)
        self.set_status(f"Rendering {start} - {end} ({n:,} frames), see the terminal for progress.")

    def set_status(self, text, warn=False):
        self.status.set_text(text)
        self.status.set_color(WARN_COLOR if warn else TEXT)
        self.fig.canvas.draw_idle()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help=".mbp-10.dbn.zst files, a folder of them, or a glob pattern")
    ap.add_argument("--fps", type=int, default=30, help="fps of the tick videos it renders")
    ap.add_argument("--out", default="lob_out", help="folder for the mp4s and cache")
    args = ap.parse_args()

    files = find_files(args.files)
    if not files:
        ap.error("no matching files")
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    viewer = Viewer(files, out_dir, args.fps)  # must stay referenced while the window is open
    plt.show()


if __name__ == "__main__":
    main()
