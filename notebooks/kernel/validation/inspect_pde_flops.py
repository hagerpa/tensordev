import pandas as pd
import numpy as np

df = pd.read_pickle('validation_outputs/kernel_flop_scaling_medium.pkl')
sub = df[df['scheme']=='heun'].copy()
sub['pde_xla_flops'] = pd.to_numeric(sub['pde_xla_flops'], errors='coerce')

print('pde_xla_flops for varying J at fixed R=9:')
rdf = sub[sub['R']==9][['J','pde_xla_flops']].drop_duplicates().sort_values('J')
print(rdf.to_string())

print()
print('pde_xla_flops for varying R (first row per R):')
for R in sorted(sub['R'].unique()):
    row = sub[sub['R']==R].iloc[0]
    flops = row['pde_xla_flops']
    print(f'  R={R}: pde_xla_flops={flops:.0f}  ratio_to_R9={(flops/27609):.3f}  (R/9)^2={(R/9)**2:.3f}  (R/9)^1={(R/9):.3f}')

print()
# Check if multiplying by J^2 gives J^2 R^2 scaling
print('pde_xla_flops * J^2 (should scale as R^2 if pde is one scan body):')
sub['pde_scaled'] = sub['pde_xla_flops'] * sub['J']**2
for R in sorted(sub['R'].unique()):
    g = sub[sub['R']==R]['pde_scaled']
    print(f'  R={R}: min={g.min():.3e}  max={g.max():.3e}  cv={g.std()/g.mean():.3f}')

