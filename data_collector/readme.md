# Data collection

- If it runs on a linux server, start Xorg first for the ithor rendering:
```bash
export DISPLAY=:0
sudo python startx.py
```

- `supervisor.py` is a keyboard debugger, nothing important to data collection
- In `single_obj.py`, set `objectType` for the target object, set `scene` for the target scene in which you collect data. `change_pos_time` is the number of random seed, and `out_dir` for the output directory. Then run:
```bash
python single_obj.py
```
