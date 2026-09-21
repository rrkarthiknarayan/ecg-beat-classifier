"""Download the MIT-BIH Arrhythmia Database (~100 MB) from PhysioNet.   Run:  python download_data.py"""
from pathlib import Path
import wfdb

dest = Path(__file__).resolve().parent / "data" / "mitdb"
dest.mkdir(parents=True, exist_ok=True)
wfdb.dl_database("mitdb", str(dest))      # all 48 records; the code skips the paced ones itself
print("saved to", dest)
