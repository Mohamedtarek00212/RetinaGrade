"""Copy a few real raw fundus images per grade into Presentation/SampleImages/.

Read-only w.r.t. the source repo: images are copied with shutil.copy2, never
modified or moved. Selection is deterministic (sorted id_code order) so the set
is reproducible. Grade labels come from the real test-split CSV.
"""
from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"d:\RetinaGrade")
PRES = ROOT / "Presentation"
OUT = PRES / "SampleImages"
SPLIT_CSV = ROOT / "data" / "splits" / "test.csv"
RAW_TEST = ROOT / "data" / "raw" / "test"

PER_GRADE = 3
CLASS_SHORT = ["NoDR", "Mild", "Moderate", "Severe", "PDR"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    by_grade: dict[int, list[str]] = defaultdict(list)
    with open(SPLIT_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_grade[int(row["diagnosis"])].append(row["id_code"])

    copied, missing = [], []
    for g in range(5):
        ids = sorted(by_grade.get(g, []))
        taken = 0
        for idc in ids:
            if taken >= PER_GRADE:
                break
            src = RAW_TEST / f"{idc}.png"
            if not src.is_file():
                continue
            dst = OUT / f"grade{g}_{CLASS_SHORT[g]}_{idc}.png"
            shutil.copy2(src, dst)
            copied.append(dst.name)
            taken += 1
        if taken == 0:
            missing.append(f"grade {g} ({CLASS_SHORT[g]})")

    print(f"COPIED {len(copied)} sample images into Presentation/SampleImages/:")
    for name in copied:
        print(f"  {name}")
    if missing:
        print("\nGrades with no available raw sample:")
        for m in missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
