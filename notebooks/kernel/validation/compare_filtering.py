#!/usr/bin/env python
"""Compare filtering effectiveness of old vs new reference parameters."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

def _kernel_workload():
    def fn(J, R, dyadic):
        J_eff = J * (2.0 ** dyadic)
        return (J_eff ** 2) * R * np.log(1 + R)
    return fn

# Old SMALL ref_point
old_ref = {'J': 64, 'R': 4, 'dyadic': 2}
# New SMALL ref_point
new_ref = {'J': 48, 'R': 3, 'dyadic': 1}

ranges = {'J': (16, 128), 'R': (2, 8), 'dyadic': (0, 4)}

# Compute full grid size
full_grid = (128-16+1) * (8-2+1) * (4-0+1)
print(f'Full grid size: {full_grid}')

# Compute reference costs
wl_fn = _kernel_workload()
old_cost = wl_fn(np.array([old_ref['J']]), np.array([old_ref['R']]), np.array([old_ref['dyadic']]))[0]
new_cost = wl_fn(np.array([new_ref['J']]), np.array([new_ref['R']]), np.array([new_ref['dyadic']]))[0]

print(f'Old ref cost: {old_cost:.2f}')
print(f'New ref cost: {new_cost:.2f}')
print(f'Cost reduction: {(1 - new_cost/old_cost)*100:.1f}%')

# Count how many combos pass each filter
J_vals = np.arange(16, 129)
R_vals = np.arange(2, 9)
D_vals = np.arange(0, 5)

count_old = 0
count_new = 0
for j in J_vals:
    for r in R_vals:
        for d in D_vals:
            cost = wl_fn(np.array([j]), np.array([r]), np.array([d]))[0]
            if cost <= old_cost:
                count_old += 1
            if cost <= new_cost:
                count_new += 1

print(f'Old filter passes: {count_old} / {full_grid} ({count_old/full_grid*100:.1f}%)')
print(f'New filter passes: {count_new} / {full_grid} ({count_new/full_grid*100:.1f}%)')
print(f'Filtered out: {count_old - count_new} more configs ({(count_old-count_new)/full_grid*100:.1f}%)')

