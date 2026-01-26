import pandas as pd

path = "/root/autodl-tmp/cafa6/helpers/real_targets/biological_process/part_08.parquet"

df = pd.read_parquet(path)

print(df.shape)
print(df.tail())

import joblib

path = "/root/autodl-tmp/cafa6/helpers/real_targets/biological_process/prior.pkl"
path = "/root/autodl-tmp/cafa6/helpers/real_targets/biological_process/nulls.pkl"
prior = joblib.load(path)

print(type(prior))
print(prior.shape)
print(prior[:10])

import glob
import pandas as pd

rt_dir = os.path.join(helpers_path, 'real_targets')
one_file = glob.glob(os.path.join(rt_dir, "**/*.parquet"), recursive=True)[0]
print("sample parquet:", one_file)

df0 = pd.read_parquet(one_file, engine="pyarrow")  # 只读一小份也行
parquet_cols = set(df0.columns)

missing = [c for c in cols if c not in parquet_cols]
print("cols missing in parquet:", len(missing))
print("example missing:", missing[:20])