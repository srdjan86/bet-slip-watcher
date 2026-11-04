# Golden review screenshots

Organized for **manual review** only. Pytest does **not** infer bet vs no-bet from these folders — it runs the same detection pipeline as the live watcher and compares to `tests/golden_manifest.yaml`.

| Folder | Meaning |
|--------|---------|
| `no-bet/` | Should **not** send Telegram (`bet_placing` / `bet_placed`) |
| `bet/placing/` | Should notify **bet submitting** (spinner) |
| `bet/placed/` | Should notify **bet placed** |

To add a frame: drop the image in the right folder, add a manifest entry (path relative to this directory), run `pytest tests/test_golden_notify.py -q`.

After moving files between folders, refresh the manifest:

```bash
python tests/sync_golden_manifest.py
pytest tests/test_golden_notify.py -q
```

If you move a file between folders, update its manifest path; `test_golden_folder_layout` fails when folder and manifest disagree.
