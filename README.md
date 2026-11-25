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

Videos longer than 30 minutes are refused (it tells you what range or `--fps` would fit); `--force` overrides.
Decoded data is cached in `lob_out/`, so re-rendering skips decoding.
