import pandas as pd
from pathlib import Path

# Paths (working paths)
DATA_DIR = Path("data")  # folder where dcgm.csv and scheduler_data.csv are
OUTPUT_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DCGM_FILE = DATA_DIR / "dcgm.csv"
SCHEDULER_FILE = DATA_DIR / "scheduler_data.csv"
CLEANED_FILE = OUTPUT_DIR / "cleaned_features.csv"

# 1. Load datasets
dcgm = pd.read_csv(DCGM_FILE)
scheduler = pd.read_csv(SCHEDULER_FILE)

# 2. Merge datasets on job id
df = pd.merge(dcgm, scheduler, left_on='id_job', right_on='id_job', how='inner')

# 3. Create features
df['overclock_proxy'] = (df['avgsmutilization_pct'] > 50).astype(int)
df['usage_hours'] = df['totalexecutiontime_sec'] / 3600
df['avg_power_watts'] = df['powerusage_watts_avg']
df['peak_power_watts'] = df['powerusage_watts_max']
df['avg_sm_pct'] = df['smutilization_pct_avg']
df['avg_mem_pct'] = df['memoryutilization_pct_avg']
df['thermal_score'] = df['powerusage_watts_max'] * df['smutilization_pct_max'] / 1000

# Select features
features = ['overclock_proxy','usage_hours','avg_power_watts','peak_power_watts',
            'avg_sm_pct','avg_mem_pct','thermal_score']

df_features = df[features]

# 4. Save cleaned features
df_features.to_csv(CLEANED_FILE, index=False)
print(f"Saved cleaned features to {CLEANED_FILE}")
print(df_features.describe())
