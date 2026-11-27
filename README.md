# LOB visualizer

This is aimed to be an app that takes limit order book input, and visualize it so you can eye-ball it like a real nerd

## Usage

```
pip install -r requirements.txt
```

**Overview**: one video per file, one frame per 5 seconds of market time:
```
python render_book.py "D:\your\folder\xnas-itch-20261026.mbp-10.dbn.zst"
python render_book.py "D:\your\folder"
```

**Tick by tick**: one frame per book update, for a range you spotted in the overview:
```
python render_book.py "D:\your\folder" --date 20261028 --tick --start 10:47:00 --end 10:48:00
```

**Executable**: double-click `build.bat` on Windows once. It creates `dist\LOBViewer\LOBViewer.exe`.
Start that (or a shortcut to it); it asks for your data folder and remembers it. Keep the whole
`dist\LOBViewer` folder together, since the exe needs the `_internal` folder next to it.

**Viewer**: pick a range on the day's chart and play it back tick by tick, live, no rendering wait:
```
python viewer.py "D:\your\folder"
```
- top chart: drag the yellow band's edges to choose a range, double-click to move it, scroll to zoom, then **Load range**
- or type the range in **From / To**: a time (`10:47:35`, `10:47:35.827614123`) or an update ID (`#1234567`); both ends are included
- every book update has an ID: its number counted from the first update in the file, so an ID always finds the same update.
  `[` / `]` set From / To to the update on screen, so you can narrow down to an exact sequence and reload it
- space = play/pause, left/right = one update, shift + arrow = 100 updates, t / shift+t = next / previous trade, home/end
- speed is either updates per second, or a multiple of real market time (1 = real speed)
- click the range chart or drag the slider to jump
- type an update ID in the `update #` box (or press `G`) and Enter to jump to it; an ID outside the loaded range loads the 1,000 updates either side of it
- **Export MP4** renders exactly the loaded updates to a video
- **Export data** opens a side panel to save tick data for any range (IDs or times, not only the loaded one)
  as CSV or Parquet, to a folder you choose. Columns: update ID, nanosecond time, mid, spread, plus your choice of
  book levels, trades, changed level and venue sequence. An optional `.txt` info file next to it records the source,
  range, columns and your note

Tick videos show each frame's update ID and nanosecond timestamp, the same IDs the viewer uses.

Videos longer than 30 minutes are refused (it tells you what range or `--fps` would fit); `--force` overrides.
Decoded data is cached in `lob_out/` (the viewer and exe put it inside your data folder), so re-rendering skips decoding.
