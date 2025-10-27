import pandas as pd
import numpy as np

def make_cluster(low, high, n, oc):
    return pd.DataFrame({
        "overclock_proxy": np.random.choice(oc, n),
        "usage_hours": np.random.uniform(low[0], high[0], n),
        "avg_power_watts": np.random.uniform(low[1], high[1], n),
        "peak_power_watts": np.random.uniform(low[2], high[2], n),
        "avg_sm_pct": np.random.uniform(low[3], high[3], n),
        "avg_mem_pct": np.random.uniform(low[4], high[4], n),
        "thermal_score": np.random.uniform(low[5], high[5], n)
    })

c0 = make_cluster([0.01,20,60,5,1,1], [1,80,150,30,15,10], 40, [0])
c1 = make_cluster([1,40,100,30,10,10], [200,160,250,70,40,22], 40, [0,1])
c2 = make_cluster([200,100,200,60,30,20], [800,240,340,100,90,35], 40, [1])

df = pd.concat([c0, c1, c2]).reset_index(drop=True)
df.to_csv("sample.csv", index=False)

print("✅ sample.csv created with correct column order.")
