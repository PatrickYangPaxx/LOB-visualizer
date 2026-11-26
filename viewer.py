"""Pick a range from a day's book and play it back tick by tick, live.

    python viewer.py "D:/data/folder"

Top chart: the day's mid price. Drag the yellow band's edges to choose a range (double-click moves it),
then Load. Playback: space = play/pause, left/right = step one update, shift+left/right = 100 updates,
t / shift+t = next / previous trade, home/end = jump to the start/end of the range.
"""
import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from lob_data import ASK_PX, ASK_SZ, BID_PX, BID_SZ, TICK, TZ
from render_book import ASK_COLOR, BID_COLOR, HALF_WINDOW, MUTED, TEXT, day_of, find_files, seconds_for, ticks_for
from render_book import main as render_main

FROZEN = getattr(sys, "frozen", False)  # running as the PyInstaller exe
APP_DIR = Path(sys.executable if FROZEN else __file__).resolve().parent

SELECT_COLOR = (242, 166, 0, 70)
MAX_LOAD = 1_000_000  # updates; beyond this the decoded range gets too big to hold comfortably


def clock(sec):
    return f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{int(sec % 60):02d}"


class ClockAxis(pg.AxisItem):
    """x values are seconds since midnight New York; ticks land on round clock times."""
    STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]

    def tickSpacing(self, minVal, maxVal, size):
        span = maxVal - minVal
        step = next((s for s in self.STEPS if span / s <= 8), 3600)
        return [(step, 0)]

    def tickStrings(self, values, scale, spacing):
        return [clock(v) if spacing < 60 else clock(v)[:5] for v in values]


class Viewer(QtWidgets.QMainWindow):
    finished = QtCore.Signal(object, object)  # (callback, result) handed back from a worker thread

    def __init__(self, files, out_dir):
        super().__init__()
        self.files, self.out_dir = files, out_dir
        self.ticks = None
        self.busy = False
        self.playing = False
        self.finished.connect(lambda callback, result: callback(result))
        self.setWindowTitle("LOB viewer")
        self.resize(1500, 900)
        self.build()

        self.timer = QtCore.QTimer(self, interval=16, timerType=QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.on_timer)
        for keys, fn in [("Space", self.toggle_play), ("Right", lambda: self.step(1)), ("Left", lambda: self.step(-1)),
                         ("Shift+Right", lambda: self.step(100)), ("Shift+Left", lambda: self.step(-100)),
                         ("T", lambda: self.jump_trade(1)), ("Shift+T", lambda: self.jump_trade(-1)),
                         ("Home", lambda: self.seek(0)), ("End", lambda: self.seek(len(self.ticks) - 1))]:
            QtGui.QShortcut(QtGui.QKeySequence(keys), self, fn)
        self.load_day(0)

    def build(self):
        pg.setConfigOptions(background="w", foreground=MUTED, antialias=True)
        central = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        top = QtWidgets.QHBoxLayout()
        self.day_box = QtWidgets.QComboBox()
        self.day_box.addItems([f.name for f in self.files])
        self.day_box.activated.connect(self.load_day)
        self.estimate = QtWidgets.QLabel()
        self.load_btn = QtWidgets.QPushButton("Load range")
        self.load_btn.clicked.connect(self.load_range)
        self.export_btn = QtWidgets.QPushButton("Export MP4")
        self.export_btn.clicked.connect(self.export)
        self.status = QtWidgets.QLabel()
        for w in (self.day_box, self.estimate, self.load_btn, self.export_btn):
            top.addWidget(w)
        top.addStretch()
        top.addWidget(self.status)
        root.addLayout(top)

        self.day_plot = pg.PlotWidget(axisItems={"bottom": ClockAxis("bottom")})
        self.day_plot.setMaximumHeight(220)
        self.day_plot.setMouseEnabled(x=True, y=False)
        self.day_plot.setAutoVisible(y=True)
        self.day_plot.setLabel("left", "Mid (USD)")
        self.day_plot.setTitle("Day: drag the band's edges to pick a range, double-click to move it, "
                               "scroll to zoom", color=MUTED, size="9pt")
        self.day_line = self.day_plot.plot(pen=pg.mkPen(TEXT, width=1))
        self.region = pg.LinearRegionItem(brush=SELECT_COLOR, pen=pg.mkPen((242, 166, 0), width=2))
        self.region.sigRegionChanged.connect(self.update_estimate)
        self.day_plot.addItem(self.region)
        self.day_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen(MUTED, width=1))
        self.day_plot.addItem(self.day_cursor)
        self.day_plot.scene().sigMouseClicked.connect(self.on_day_click)
        root.addWidget(self.day_plot)

        labels = QtWidgets.QHBoxLayout()
        mono = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        mono.setPointSize(13)
        self.clock_lbl = QtWidgets.QLabel()
        self.clock_lbl.setFont(mono)
        mono.setPointSize(10)
        self.info_lbl = QtWidgets.QLabel()
        self.info_lbl.setFont(mono)
        labels.addWidget(self.clock_lbl)
        labels.addStretch()
        labels.addWidget(self.info_lbl)
        root.addLayout(labels)

        plots = QtWidgets.QHBoxLayout()
        self.ladder = pg.PlotWidget()
        self.ladder.setMouseEnabled(False, False)
        self.ladder.hideButtons()
        self.ladder.setLabel("left", "Price (USD)")
        self.ladder.setLabel("bottom", "Order size at level (shares)")
        self.ladder.getAxis("left").tickStrings = lambda values, scale, spacing: [f"{v:.2f}" for v in values]
        self.ladder.showGrid(x=True, alpha=0.15)
        bar_h = TICK * 0.8
        self.bids = pg.BarGraphItem(x0=np.zeros(10), y=np.zeros(10), height=bar_h, width=np.zeros(10),
                                    brush=BID_COLOR, pen=None)
        self.asks = pg.BarGraphItem(x0=np.zeros(10), y=np.zeros(10), height=bar_h, width=np.zeros(10),
                                    brush=ASK_COLOR, pen=None)
        self.outline = pg.BarGraphItem(x0=[0], y=[0], height=bar_h, width=[0], brush=None,
                                       pen=pg.mkPen(TEXT, width=2))
        self.trade_mark = pg.ScatterPlotItem(symbol="t2", size=14, brush=TEXT, pen=None)
        self.mid_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(MUTED, width=1, style=QtCore.Qt.DashLine))
        for item in (self.bids, self.asks, self.outline, self.trade_mark, self.mid_line):
            self.ladder.addItem(item)
        plots.addWidget(self.ladder, 1)

        self.range_plot = pg.PlotWidget()
        self.range_plot.setLabel("left", "Mid (USD)")
        self.range_plot.setLabel("bottom", "Book update #")
        self.range_plot.setTitle("Loaded range: click to jump", color=MUTED, size="9pt")
        self.range_plot.showGrid(x=True, y=True, alpha=0.15)
        self.range_line = self.range_plot.plot(pen=pg.mkPen(TEXT, width=1))
        self.range_cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen(MUTED, width=1))
        self.range_dot = pg.ScatterPlotItem(size=10, brush=TEXT, pen=pg.mkPen("w", width=2))
        self.range_plot.addItem(self.range_cursor)
        self.range_plot.addItem(self.range_dot)
        self.range_plot.scene().sigMouseClicked.connect(self.on_range_click)
        plots.addWidget(self.range_plot, 1)
        root.addLayout(plots, 1)

        controls = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        buttons = [self.play_btn]
        for label, fn in [("< step", lambda: self.step(-1)), ("step >", lambda: self.step(1)),
                          ("< trade", lambda: self.jump_trade(-1)), ("trade >", lambda: self.jump_trade(1))]:
            b = QtWidgets.QPushButton(label)
            b.clicked.connect(fn)
            buttons.append(b)
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(["updates / s", "x market time"])
        self.mode_box.activated.connect(self.on_mode)
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(0.01, 100_000)
        self.speed.setValue(30)
        self.speed.setFixedWidth(100)
        self.pos_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.pos_slider.valueChanged.connect(lambda v: self.seek(v) if v != getattr(self, "i", None) else None)
        # keep space and arrows for playback instead of whichever control has focus
        for w in buttons + [self.mode_box, self.pos_slider, self.day_box, self.load_btn, self.export_btn]:
            w.setFocusPolicy(QtCore.Qt.NoFocus)
        for w in buttons + [QtWidgets.QLabel("Speed"), self.speed, self.mode_box]:
            controls.addWidget(w)
        controls.addWidget(self.pos_slider, 1)
        root.addLayout(controls)
        self.set_playback_enabled(False)

    def run_in_background(self, fn, callback, message):
        self.busy = True
        self.status.setText(message)

        def work():
            try:
                result = fn()
            except Exception as e:  # shown in the status line instead of killing the thread silently
                result = e
            self.finished.emit(callback, result)

        threading.Thread(target=work, daemon=True).start()

    def load_day(self, day_i):
        if self.busy:
            return
        path = self.files[day_i]
        self.day_box.setCurrentIndex(day_i)
        self.run_in_background(lambda: seconds_for(path, self.out_dir), lambda bars: self.day_loaded(path, bars),
                               f"Loading {path.name} (first time decodes the whole file) ...")

    def day_loaded(self, path, bars):
        self.busy = False
        if isinstance(bars, Exception):
            self.status.setText(f"Failed: {bars}")
            return
        self.path = path
        self.midnight = pd.Timestamp(day_of(path), tz=TZ)
        bars = bars.between_time("09:30", "16:00")
        self.day_sec = (bars.index - self.midnight).total_seconds().to_numpy()
        self.day_updates = bars["updates"].to_numpy()
        self.day_line.setData(self.day_sec, ((bars["bid_px_00"] + bars["ask_px_00"]) / 2).to_numpy())
        open_ = 9.5 * 3600
        self.day_plot.setXRange(open_, 16 * 3600, padding=0.01)
        self.region.setRegion([open_, open_ + 60])
        self.status.setText(f"{path.name}: pick a range and Load it.")

    def on_day_click(self, ev):
        if not ev.double() or not hasattr(self, "day_sec"):
            return
        x = self.day_plot.getViewBox().mapSceneToView(ev.scenePos()).x()
        lo, hi = self.region.getRegion()
        self.region.setRegion([x, x + hi - lo])

    def selected(self):
        lo, hi = self.region.getRegion()
        return int(lo), int(np.ceil(hi))

    def update_estimate(self):
        if not hasattr(self, "day_sec"):
            return
        lo, hi = self.selected()
        n = int(self.day_updates[(self.day_sec >= lo) & (self.day_sec < hi)].sum())
        self.estimate.setText(f"  {clock(lo)} - {clock(hi)}  |  {n:,} updates  ")
        self.estimate.setStyleSheet("color: #ec4647" if n > MAX_LOAD else "")

    def load_range(self):
        if self.busy or not hasattr(self, "day_sec"):
            return
        lo, hi = self.selected()
        n = int(self.day_updates[(self.day_sec >= lo) & (self.day_sec < hi)].sum())
        if n > MAX_LOAD:
            self.status.setText(f"{n:,} updates is too many to load at once (max {MAX_LOAD:,}). Narrow the range.")
            return
        start, end = self.midnight + pd.Timedelta(seconds=lo), self.midnight + pd.Timedelta(seconds=hi)
        self.pause()
        self.loading_span = (clock(lo), clock(hi))
        self.run_in_background(lambda: ticks_for(self.path, start, end, self.out_dir), self.range_loaded,
                               f"Decoding {clock(lo)} - {clock(hi)} ...")

    def range_loaded(self, ticks):
        self.busy = False
        if isinstance(ticks, Exception):
            self.status.setText(f"Failed: {ticks}")
            return
        if ticks.empty:
            self.status.setText("No book updates in that range.")
            return
        self.ticks = ticks
        self.bid_px, self.ask_px = ticks[BID_PX].to_numpy(), ticks[ASK_PX].to_numpy()
        self.bid_sz, self.ask_sz = ticks[BID_SZ].to_numpy(), ticks[ASK_SZ].to_numpy()
        self.mid = (self.bid_px[:, 0] + self.ask_px[:, 0]) / 2
        self.change_px = ticks["change_px"].to_numpy(float)
        self.change_side = ticks["change_side"].to_numpy()
        self.trade_px = ticks["trade_px"].to_numpy(float)
        self.buy, self.sell = ticks["buy_vol"].to_numpy(), ticks["sell_vol"].to_numpy()
        self.trades = np.flatnonzero(~np.isnan(self.trade_px))
        self.tick_sec = (ticks.index - self.midnight).total_seconds().to_numpy()
        self.labels = [t.strftime("%Y-%m-%d  %H:%M:%S.%f  New York") for t in ticks.index]
        self.range_span = self.loading_span  # same start/end as the cached decode, so export reuses it

        n = len(ticks)
        xmax = np.percentile(np.concatenate([self.bid_sz, self.ask_sz]), 99.5) * 1.1
        self.ladder.setXRange(0, xmax, padding=0)
        self.range_line.setData(np.arange(n), self.mid)
        self.range_plot.autoRange()
        self.pos_slider.setRange(0, n - 1)
        self.set_playback_enabled(True)
        self.status.setText(f"Loaded {n:,} updates, {len(self.trades):,} with trades. Space to play.")
        self.seek(0)

    def set_playback_enabled(self, on):
        for w in self.findChildren(QtWidgets.QPushButton) + [self.speed, self.mode_box, self.pos_slider]:
            if w is not self.load_btn:
                w.setEnabled(on)

    def show_frame(self, i):
        self.i = i
        bid_px, ask_px, bid_sz, ask_sz = self.bid_px[i], self.ask_px[i], self.bid_sz[i], self.ask_sz[i]
        self.bids.setOpts(y=bid_px, width=bid_sz)
        self.asks.setOpts(y=ask_px, width=ask_sz)
        mid = self.mid[i]
        self.mid_line.setValue(mid)
        self.ladder.setYRange(mid - HALF_WINDOW, mid + HALF_WINDOW, padding=0)

        # outline the level this update touched
        side, px = self.change_side[i], self.change_px[i]
        levels, sizes = (bid_px, bid_sz) if side == "B" else (ask_px, ask_sz)
        hit = np.flatnonzero(np.abs(levels - px) < TICK / 2) if side in ("B", "A") else []
        self.outline.setVisible(len(hit) > 0)
        if len(hit):
            self.outline.setOpts(y=[levels[hit[0]]], width=[sizes[hit[0]]])
        if np.isnan(self.trade_px[i]):
            self.trade_mark.clear()
        else:
            self.trade_mark.setData([self.ladder.viewRange()[0][1] * 0.015], [self.trade_px[i]])

        spread = round((ask_px[0] - bid_px[0]) / TICK)
        self.clock_lbl.setText(self.labels[i])
        self.info_lbl.setText(f"spread {spread} tick{'s' if spread != 1 else ''}   |   "
                              f"update {i + 1:,}/{len(self.ticks):,}   |   "
                              f"buy {int(self.buy[i]):,} sh   sell {int(self.sell[i]):,} sh")
        self.range_cursor.setValue(i)
        self.range_dot.setData([i], [mid])
        self.day_cursor.setValue(self.tick_sec[i])
        self.pos_slider.setValue(i)

    def seek(self, i):
        if self.ticks is None:
            return
        i = int(np.clip(i, 0, len(self.ticks) - 1))
        self.pos, self.clock_sec = float(i), self.tick_sec[i]
        self.show_frame(i)

    def step(self, n):
        if self.ticks is not None:
            self.pause()
            self.seek(self.i + n)

    def jump_trade(self, direction):
        if self.ticks is None or not len(self.trades):
            return
        k = np.searchsorted(self.trades, self.i, side="right" if direction > 0 else "left") + min(direction, 0)
        if 0 <= k < len(self.trades):
            self.pause()
            self.seek(self.trades[k])

    def on_range_click(self, ev):
        if self.ticks is not None:
            self.step(round(self.range_plot.getViewBox().mapSceneToView(ev.scenePos()).x()) - self.i)

    def on_mode(self, mode):
        self.speed.setValue(30 if mode == 0 else 1)
        if self.ticks is not None:
            self.seek(self.i)

    def toggle_play(self):
        if self.ticks is None:
            return
        if self.playing:
            self.pause()
            return
        if self.i >= len(self.ticks) - 1:
            self.seek(0)
        self.playing = True
        self.play_btn.setText("Pause")
        self.last = time.perf_counter()
        self.timer.start()

    def pause(self):
        self.playing = False
        self.play_btn.setText("Play")
        self.timer.stop()

    def on_timer(self):
        now = time.perf_counter()
        dt, self.last = now - self.last, now
        if self.mode_box.currentIndex() == 0:
            self.pos += self.speed.value() * dt
            i = int(self.pos)
        else:
            self.clock_sec += self.speed.value() * dt
            i = int(np.searchsorted(self.tick_sec, self.clock_sec, side="right")) - 1
        last = len(self.ticks) - 1
        if i >= last:
            self.pause()
            i = last
        if i != self.i:
            self.show_frame(max(i, 0))

    def export(self):
        if self.ticks is None:
            return
        start, end = self.range_span
        # a second copy of this program does the render, in its own console window showing progress
        cmd = [sys.executable] + ([] if FROZEN else [str(Path(__file__).resolve())]) + [
            "--render", str(self.path), "--tick", "--start", start, "--end", end, "--out", str(self.out_dir)]
        subprocess.Popen(cmd, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        self.status.setText(f"Exporting {start} - {end} to MP4, progress in the new console window.")


def render_and_wait():
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    try:
        render_main()
    except SystemExit as e:  # render_book exits with a message, e.g. over the length cap
        if e.code not in (None, 0):
            print(e.code)
    except Exception:
        import traceback
        traceback.print_exc()
    input("\nPress Enter to close")


def main():
    if sys.argv[1:2] == ["--render"]:
        render_and_wait()
        return
    if FROZEN and sys.platform == "win32":
        # the exe is a console app so exports get a progress window; the viewer itself doesn't need one
        import ctypes
        ctypes.windll.kernel32.FreeConsole()
        sys.stdout = sys.stderr = open(os.devnull, "w")

    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help=".mbp-10.dbn.zst files, a folder of them, or a glob pattern")
    ap.add_argument("--out", default=str(APP_DIR / "lob_out"), help="folder for the cache and exported mp4s")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    if args.files:
        files = find_files(args.files)
    else:
        settings = QtCore.QSettings("LOB-visualizer", "viewer")
        folder = QtWidgets.QFileDialog.getExistingDirectory(None, "Folder with .mbp-10.dbn.zst files",
                                                            settings.value("folder", ""))
        if not folder:
            return
        settings.setValue("folder", folder)
        files = find_files([folder])
    if not files:
        QtWidgets.QMessageBox.warning(None, "LOB viewer", "No .mbp-10.dbn.zst files found there.")
        return
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    viewer = Viewer(files, out_dir)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
