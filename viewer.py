"""Pick a range from a day's book and play it back tick by tick, live.

    python viewer.py "D:/data/folder"

Pick a range by dragging the yellow band on the day chart (double-click moves it), or type it in the
From / To boxes as a time (10:47:35, 10:47:35.827614123) or an update ID (#1234567), then Load.
Every book update has an ID: its number counted from the first update in the file, so the same ID
always finds the same update. Both ends of a range are included.

Playback: space = play/pause, left/right = step one update, shift+left/right = 100 updates,
t / shift+t = next / previous trade, home/end = start/end of the range,
[ / ] = set From / To to the current update.
"""
import argparse
import html
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from lob_data import ASK_PX, ASK_SZ, BID_PX, BID_SZ, TICK, TZ
from render_book import ASK_COLOR, BID_COLOR, HALF_WINDOW, MUTED, TEXT, day_of, find_files, seconds_for, ticks_for
from render_book import main as render_main

FROZEN = getattr(sys, "frozen", False)  # running as the PyInstaller exe

SELECT_COLOR = (242, 166, 0, 70)
MAX_LOAD = 1_000_000  # updates; beyond this the decoded range gets too big to hold comfortably
NS = 1_000_000_000


def clock(sec):
    return f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{int(sec % 60):02d}"


def parse_point(text):
    """'#1234567' -> ("id", 1234567, False); '10:47:35.8276' -> ("ns", ns since midnight, whole_second)."""
    t = text.strip().replace(",", "").replace(" ", "")
    bad = ValueError(f"'{text}' is not a time like 10:47:35.123 or an ID like #1234567")
    if t.startswith("#"):
        if not t[1:].isdigit():
            raise bad
        return "id", int(t[1:]), False
    hms, _, frac = t.partition(".")
    parts = hms.split(":")
    if len(parts) == 2:
        parts.append("0")
    if len(parts) != 3 or not all(p.isdigit() for p in parts) or len(frac) > 9 or (frac and not frac.isdigit()):
        raise bad
    h, m, s = map(int, parts)
    return "ns", ((h * 60 + m) * 60 + s) * NS + int(frac.ljust(9, "0")), not frac


def select(ticks, first_id, frm, to, midnight_ns):
    """Rows of a decoded window that fall inside From/To, with their update IDs attached."""
    ids = first_id + np.arange(len(ticks))
    tick_ns = ticks.index.asi8 - midnight_ns
    keep = (ids >= frm[1]) if frm[0] == "id" else (tick_ns >= frm[1])
    if to[0] == "id":
        keep &= ids <= to[1]
    else:
        # a whole second like 10:47:35 means up to the end of that second
        keep &= (tick_ns < to[1] + NS) if to[2] else (tick_ns <= to[1])
    return ticks[keep].assign(update_id=ids[keep])


def data_table(ticks, levels, trades, changes, sequence):
    """The columns written by the data export, one row per book update."""
    ns = ticks.index.asi8
    out = pd.DataFrame({
        "update_id": ticks["update_id"].to_numpy(),
        "time_ny": [f"{t:%Y-%m-%d %H:%M:%S}.{n % NS:09d}" for t, n in zip(ticks.index, ns)],
        "ts_recv_ns": ns,  # nanoseconds since 1970-01-01 UTC, for joining with other data
    })
    if sequence:
        out["sequence"] = ticks["sequence"].to_numpy()
    bid, ask = ticks["bid_px_00"].to_numpy(), ticks["ask_px_00"].to_numpy()
    out["mid"] = (bid + ask) / 2
    out["spread_ticks"] = np.round((ask - bid) / TICK).astype(int)
    for i in range(levels):
        for col in (f"bid_px_{i:02d}", f"bid_sz_{i:02d}", f"ask_px_{i:02d}", f"ask_sz_{i:02d}"):
            out[col] = ticks[col].to_numpy()
    if trades:
        for col in ("buy_vol", "sell_vol", "trade_px"):
            out[col] = ticks[col].to_numpy()
    if changes:
        out["change_side"] = ticks["change_side"].to_numpy()
        out["change_px"] = ticks["change_px"].to_numpy()
    return out


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
                         ("Home", lambda: self.seek(0)), ("End", lambda: self.seek(len(self.ticks) - 1)),
                         ("[", lambda: self.mark(self.from_box)), ("]", lambda: self.mark(self.to_box))]:
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
        self.from_box = QtWidgets.QLineEdit()
        self.to_box = QtWidgets.QLineEdit()
        for box, tip in [(self.from_box, "first update"), (self.to_box, "last update")]:
            box.setPlaceholderText("HH:MM:SS.fff or #ID")
            box.setToolTip(f"{tip}: a time like 10:47:35 or 10:47:35.827614123, or an update ID like #1234567")
            box.setFixedWidth(190)
            box.textEdited.connect(self.update_estimate)
            box.returnPressed.connect(self.load_range)
        self.estimate = QtWidgets.QLabel()
        self.load_btn = QtWidgets.QPushButton("Load range")
        self.load_btn.clicked.connect(self.load_range)
        self.export_btn = QtWidgets.QPushButton("Export MP4")
        self.export_btn.clicked.connect(self.export)
        self.data_btn = QtWidgets.QPushButton("Export data")
        self.data_btn.setCheckable(True)
        self.status = QtWidgets.QLabel()
        for w in (self.day_box, QtWidgets.QLabel("From"), self.from_box, QtWidgets.QLabel("To"), self.to_box,
                  self.load_btn, self.estimate, self.export_btn, self.data_btn):
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
        self.region.sigRegionChanged.connect(self.on_region)
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
        for lbl in (self.clock_lbl, self.info_lbl):  # so IDs and timestamps can be copied
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
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
        self.range_plot.setLabel("bottom", "Position in loaded range")
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
                          ("< trade", lambda: self.jump_trade(-1)), ("trade >", lambda: self.jump_trade(1)),
                          ("From = here [", lambda: self.mark(self.from_box)),
                          ("To = here ]", lambda: self.mark(self.to_box))]:
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
        for w in buttons + [self.mode_box, self.pos_slider, self.day_box, self.load_btn, self.export_btn,
                            self.data_btn]:
            w.setFocusPolicy(QtCore.Qt.NoFocus)
        for w in buttons + [QtWidgets.QLabel("Speed"), self.speed, self.mode_box]:
            controls.addWidget(w)
        controls.addWidget(self.pos_slider, 1)
        root.addLayout(controls)
        self.playback_widgets = buttons + [self.speed, self.mode_box, self.pos_slider, self.export_btn]
        self.set_playback_enabled(False)
        self.build_data_panel()

    def build_data_panel(self):
        self.dock = QtWidgets.QDockWidget("Export tick data", self)
        self.dock.setObjectName("data_panel")
        panel = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(panel)
        form = QtWidgets.QFormLayout()
        outer.addLayout(form)
        outer.addStretch()

        self.x_from, self.x_to = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for label, box in [("From", self.x_from), ("To", self.x_to)]:
            box.setPlaceholderText("HH:MM:SS.fff or #ID")
            box.textEdited.connect(self.update_data_count)
            here = QtWidgets.QPushButton("here")
            here.setToolTip("the update on screen")
            here.clicked.connect(lambda _=False, b=box: self.mark(b))
            row = QtWidgets.QHBoxLayout()
            row.addWidget(box, 1)
            row.addWidget(here)
            form.addRow(label, row)
        use_loaded = QtWidgets.QPushButton("Use loaded range")
        use_loaded.clicked.connect(self.use_loaded_range)
        form.addRow("", use_loaded)
        self.x_count = QtWidgets.QLabel()
        form.addRow("", self.x_count)

        self.x_levels = QtWidgets.QSpinBox()
        self.x_levels.setRange(1, 10)
        self.x_levels.setValue(10)
        form.addRow("Book levels", self.x_levels)
        self.x_trades = QtWidgets.QCheckBox("Trades: buy / sell volume, trade price")
        self.x_changes = QtWidgets.QCheckBox("Changed level: side, price")
        self.x_seq = QtWidgets.QCheckBox("Venue sequence number")
        for box in (self.x_trades, self.x_changes, self.x_seq):
            box.setChecked(True)
            form.addRow("", box)
        always = QtWidgets.QLabel("Always included: update ID, time, mid, spread")
        always.setWordWrap(True)
        form.addRow("", always)

        self.x_note = QtWidgets.QPlainTextEdit()
        self.x_note.setPlaceholderText("Optional, saved in the info file")
        self.x_note.setFixedHeight(70)
        form.addRow("Note", self.x_note)
        self.x_info = QtWidgets.QCheckBox("Save an info file (.txt) next to the data")
        self.x_info.setChecked(True)
        form.addRow("", self.x_info)

        self.x_btn = QtWidgets.QPushButton("Export...")
        self.x_btn.clicked.connect(self.export_data)
        form.addRow("", self.x_btn)
        self.x_status = QtWidgets.QLabel()
        self.x_status.setWordWrap(True)
        self.x_status.setTextFormat(QtCore.Qt.RichText)
        self.x_status.setOpenExternalLinks(True)
        form.addRow("", self.x_status)
        for w in panel.findChildren(QtWidgets.QAbstractButton):
            w.setFocusPolicy(QtCore.Qt.NoFocus)

        panel.setFixedWidth(330)
        self.dock.setWidget(panel)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dock)
        self.dock.visibilityChanged.connect(self.data_btn.setChecked)
        self.data_btn.toggled.connect(self.dock.setVisible)
        self.dock.hide()

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
        self.ticks = None
        self.set_playback_enabled(False)
        # IDs count from the first update in the file, pre-market included
        self.all_sec = (bars.index - self.midnight).total_seconds().to_numpy().astype(np.int64)
        self.all_cum = bars["updates"].cumsum().to_numpy()
        bars = bars.between_time("09:30", "16:00")
        self.day_sec = (bars.index - self.midnight).total_seconds().to_numpy()
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

    def on_region(self):
        if getattr(self, "syncing", False):
            return
        lo, hi = self.region.getRegion()
        self.from_box.setText(clock(int(lo)))
        self.to_box.setText(clock(int(np.ceil(hi)) - 1))  # To is inclusive, so the last whole second
        self.update_estimate()

    def updates_before(self, sec):
        k = np.searchsorted(self.all_sec, sec)
        return int(self.all_cum[k - 1]) if k else 0

    def id_second(self, update_id):
        k = np.searchsorted(self.all_cum, update_id)
        if update_id < 1 or k >= len(self.all_cum):
            raise ValueError(f"no update #{update_id:,} in this file (last is #{int(self.all_cum[-1]):,})")
        return int(self.all_sec[k])

    def bounds(self, from_box, to_box):
        """Parsed From/To, plus the whole seconds [lo, hi) that have to be decoded to cover them."""
        frm, to = parse_point(from_box.text()), parse_point(to_box.text())
        lo = self.id_second(frm[1]) if frm[0] == "id" else frm[1] // NS
        hi = (self.id_second(to[1]) if to[0] == "id" else to[1] // NS) + 1
        if hi <= lo:
            raise ValueError("From is after To")
        return frm, to, lo, hi

    def show_count(self, label, from_box, to_box):
        if not hasattr(self, "all_sec"):
            return
        try:
            frm, to, lo, hi = self.bounds(from_box, to_box)
        except ValueError as e:
            label.setText(f"  {e}  ")
            label.setStyleSheet("color: #ec4647")
            return
        n = self.updates_before(hi) - self.updates_before(lo)
        exact = frm[0] == to[0] == "id"
        count = f"{to[1] - frm[1] + 1:,} updates" if exact else f"up to {n:,} updates"
        label.setText(f"  {count}  ")
        label.setStyleSheet("color: #ec4647" if n > MAX_LOAD else "")

    def update_estimate(self):
        self.show_count(self.estimate, self.from_box, self.to_box)

    def update_data_count(self):
        self.show_count(self.x_count, self.x_from, self.x_to)

    def load_range(self):
        focused = QtWidgets.QApplication.focusWidget()
        if focused:
            focused.clearFocus()  # hand the keyboard back to playback after typing
        if self.busy or not hasattr(self, "all_sec"):
            return
        try:
            frm, to, lo, hi = self.bounds(self.from_box, self.to_box)
        except ValueError as e:
            self.status.setText(f"Can't load: {e}")
            return
        n = self.updates_before(hi) - self.updates_before(lo)
        if n > MAX_LOAD:
            self.status.setText(f"{n:,} updates is too many to load at once (max {MAX_LOAD:,}). Narrow the range.")
            return
        self.syncing = True  # move the band without overwriting what was typed
        self.region.setRegion([lo, hi])
        self.syncing = False
        start, end = self.midnight + pd.Timedelta(seconds=lo), self.midnight + pd.Timedelta(seconds=hi)
        self.pause()
        self.pending = (frm, to, lo, hi)
        self.run_in_background(lambda: ticks_for(self.path, start, end, self.out_dir), self.range_loaded,
                               f"Decoding {clock(lo)} - {clock(hi)} ...")

    def range_loaded(self, ticks):
        self.busy = False
        if isinstance(ticks, Exception):
            self.status.setText(f"Failed: {ticks}")
            return
        frm, to, lo, hi = self.pending
        self.window = (lo, hi, ticks)  # the whole decoded seconds, so data exports inside it skip decoding
        ticks = select(ticks, self.updates_before(lo) + 1, frm, to, self.midnight.value)
        if ticks.empty:
            self.status.setText("No book updates in that range.")
            return
        self.ticks = ticks
        self.ids = ticks["update_id"].to_numpy()
        self.seq = ticks["sequence"].to_numpy()
        tick_ns = ticks.index.asi8 - self.midnight.value
        if not self.x_from.text() and not self.x_to.text():
            self.use_loaded_range()
        self.bid_px, self.ask_px = ticks[BID_PX].to_numpy(), ticks[ASK_PX].to_numpy()
        self.bid_sz, self.ask_sz = ticks[BID_SZ].to_numpy(), ticks[ASK_SZ].to_numpy()
        self.mid = (self.bid_px[:, 0] + self.ask_px[:, 0]) / 2
        self.change_px = ticks["change_px"].to_numpy(float)
        self.change_side = ticks["change_side"].to_numpy()
        self.trade_px = ticks["trade_px"].to_numpy(float)
        self.buy, self.sell = ticks["buy_vol"].to_numpy(), ticks["sell_vol"].to_numpy()
        self.trades = np.flatnonzero(~np.isnan(self.trade_px))
        self.tick_sec = tick_ns / NS
        self.labels = [f"{t:%Y-%m-%d  %H:%M:%S}.{ns % NS:09d}  New York" for t, ns in zip(ticks.index, tick_ns)]

        n = len(ticks)
        xmax = np.percentile(np.concatenate([self.bid_sz, self.ask_sz]), 99.5) * 1.1
        self.ladder.setXRange(0, xmax, padding=0)
        self.range_line.setData(np.arange(n), self.mid)
        self.range_plot.autoRange()
        self.pos_slider.setRange(0, n - 1)
        self.set_playback_enabled(True)
        self.status.setText(f"Loaded #{self.ids[0]:,} - #{self.ids[-1]:,}: {n:,} updates, "
                            f"{len(self.trades):,} with trades. Space to play.")
        self.seek(0)

    def set_playback_enabled(self, on):
        for w in self.playback_widgets:
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
        self.info_lbl.setText(f"update #{self.ids[i]:,}  ({i + 1:,}/{len(self.ticks):,})   |   "
                              f"venue seq {self.seq[i]}   |   spread {spread} tick{'s' if spread != 1 else ''}"
                              f"   |   buy {int(self.buy[i]):,} sh   sell {int(self.sell[i]):,} sh")
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

    def mark(self, box):
        if self.ticks is not None:
            box.setText(f"#{self.ids[self.i]}")
            self.update_estimate()
            self.update_data_count()

    def use_loaded_range(self):
        if self.ticks is not None:
            self.x_from.setText(f"#{self.ids[0]}")
            self.x_to.setText(f"#{self.ids[-1]}")
            self.update_data_count()

    def export_data(self):
        if self.busy or not hasattr(self, "all_sec"):
            return
        try:
            frm, to, lo, hi = self.bounds(self.x_from, self.x_to)
        except ValueError as e:
            self.x_status.setText(f"Can't export: {e}")
            return
        n = self.updates_before(hi) - self.updates_before(lo)
        if n > MAX_LOAD:
            self.x_status.setText(f"{n:,} updates is too many at once (max {MAX_LOAD:,}). Split the range.")
            return
        midnight, first_id = self.midnight.value, self.updates_before(lo) + 1
        window = getattr(self, "window", None)
        if self.ticks is not None and window and window[0] <= lo and hi <= window[1]:
            w_lo, _, raw = window
            self.data_ready(select(raw, self.updates_before(w_lo) + 1, frm, to, midnight))
            return
        start, end = self.midnight + pd.Timedelta(seconds=lo), self.midnight + pd.Timedelta(seconds=hi)
        self.x_status.setText("Decoding ...")
        self.run_in_background(lambda: select(ticks_for(self.path, start, end, self.out_dir), first_id, frm, to,
                                              midnight),
                               self.data_ready, f"Decoding {clock(lo)} - {clock(hi)} for export ...")

    def data_ready(self, ticks):
        self.busy = False
        self.status.setText("")
        if isinstance(ticks, Exception):
            self.x_status.setText(f"Failed: {ticks}")
            return
        if ticks.empty:
            self.x_status.setText("No book updates in that range.")
            return
        first, last = int(ticks["update_id"].iloc[0]), int(ticks["update_id"].iloc[-1])
        settings = QtCore.QSettings("LOB-visualizer", "viewer")
        folder = Path(settings.value("export_dir", str(self.path.parent)))
        name = f"intc_{day_of(self.path)}_{first}-{last}.csv"
        chosen, kind = QtWidgets.QFileDialog.getSaveFileName(self, "Save tick data", str(folder / name),
                                                             "CSV (*.csv);;Parquet (*.parquet)")
        if not chosen:
            self.x_status.setText("Export cancelled.")
            return
        path = Path(chosen).resolve()
        if path.suffix.lower() not in (".csv", ".parquet"):
            path = path.with_suffix(".parquet" if kind.startswith("Parquet") else ".csv")
        settings.setValue("export_dir", str(path.parent))

        options = (self.x_levels.value(), self.x_trades.isChecked(), self.x_changes.isChecked(),
                   self.x_seq.isChecked())
        write_info, note, source = self.x_info.isChecked(), self.x_note.toPlainText().strip(), self.path

        def write():
            table = data_table(ticks, *options)
            if path.suffix.lower() == ".parquet":
                table.to_parquet(path, index=False)
            else:
                table.to_csv(path, index=False)
            if write_info:
                path.with_suffix(".txt").write_text("\n".join([
                    f"data:     {path.name}",
                    f"source:   {source}",
                    f"updates:  #{first:,} - #{last:,} ({len(table):,} rows)",
                    f"time:     {table['time_ny'].iloc[0]} - {table['time_ny'].iloc[-1]} New York",
                    f"columns:  {', '.join(table.columns)}",
                    f"exported: {datetime.now():%Y-%m-%d %H:%M:%S}",
                    f"note:     {note}",
                ]) + "\n", encoding="utf-8")
            return path

        self.x_status.setText(f"Writing {len(ticks):,} rows ...")
        self.run_in_background(write, self.data_written, f"Writing {path.name} ...")

    def data_written(self, path):
        self.busy = False
        self.status.setText("")
        if isinstance(path, Exception):
            self.x_status.setText(f"Failed to write: {path}")
            return
        folder = QtCore.QUrl.fromLocalFile(str(path.parent)).toString()
        self.x_status.setText(f'Saved {html.escape(str(path))}<br><a href="{folder}">Open folder</a>')

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
        first, last = self.ids[0], self.ids[-1]
        # hand over exactly what is loaded, so the video matches update for update
        ticks_file = self.out_dir / f"{self.path.name.split('.')[0]}_{first}-{last}.export.parquet"
        self.ticks.to_parquet(ticks_file)
        # a second copy of this program does the render, in its own console window showing progress
        cmd = [sys.executable] + ([] if FROZEN else [str(Path(__file__).resolve())]) + [
            "--render", str(self.path), "--tick", "--ticks-file", str(ticks_file), "--out", str(self.out_dir)]
        subprocess.Popen(cmd, creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
        self.status.setText(f"Exporting #{first:,} - #{last:,} to MP4, progress in the new console window.")


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
    ap.add_argument("--out", help="folder for the cache and exported mp4s (default: lob_out in the data folder)")
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
    # next to the data, not the exe: rebuilding the exe wipes its folder
    out_dir = Path(args.out) if args.out else files[0].parent / "lob_out"
    out_dir.mkdir(exist_ok=True)
    viewer = Viewer(files, out_dir)
    viewer.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
