"""Print a compact profile of every CSV under the given dataset folders."""
import sys, glob, os
import pandas as pd

pd.set_option("display.width", 220)
for root in sys.argv[1:]:
    for path in sorted(glob.glob(os.path.join(root, "data", "**", "*.csv"), recursive=True)):
        df = pd.read_csv(path, low_memory=False)
        print(f"\n######## {os.path.relpath(path, root)}  rows={len(df):,} cols={df.shape[1]} dup_rows={df.duplicated().sum():,}")
        for c in df.columns:
            s = df[c]
            nun = s.nunique(dropna=True)
            miss = s.isna().mean() * 100
            if pd.api.types.is_numeric_dtype(s):
                info = f"min={s.min()} max={s.max()} mean={s.mean():.4g}"
            else:
                top = s.astype(str).value_counts().head(3)
                info = "top=" + "; ".join(f"{k[:40]!r}x{v}" for k, v in top.items())
            print(f"  {c[:28]:28s} {str(s.dtype):8s} uniq={nun:<7} miss={miss:5.1f}%  {info}")
