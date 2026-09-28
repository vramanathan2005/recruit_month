# Commitment Strength Holdout Validation

- Known commitment-event outcomes: 12883
- Train sample: classes 2022-2024
- Holdout sample: class 2025
- Live/current-ish sample: class 2026, interpret carefully because outcomes can still be evolving

| Sample | Known Events | Decommit/Flip Rate | AUC |
| --- | ---: | ---: | ---: |
| train_2022_2024 | 6374 | 23.1% | 0.7224 |
| holdout_2025 | 3088 | 23.3% | 0.6982 |
| live_2026 | 3207 | 24.6% | 0.6957 |

| Sample | Bucket | Known Events | Decommit/Flip Rate | Stick Rate |
| --- | --- | ---: | ---: | ---: |
| train_2022_2024 | Weak / Attack | 360 | 49.2% | 50.8% |
| train_2022_2024 | Vulnerable | 2572 | 34.7% | 65.3% |
| train_2022_2024 | Stable / Monitor | 1794 | 17.2% | 82.8% |
| train_2022_2024 | Strong | 1648 | 5.9% | 94.1% |
| holdout_2025 | Weak / Attack | 198 | 52.0% | 48.0% |
| holdout_2025 | Vulnerable | 1341 | 30.9% | 69.0% |
| holdout_2025 | Stable / Monitor | 826 | 18.4% | 81.6% |
| holdout_2025 | Strong | 723 | 6.8% | 93.2% |
| live_2026 | Weak / Attack | 190 | 54.7% | 45.3% |
| live_2026 | Vulnerable | 1434 | 32.4% | 67.6% |
| live_2026 | Stable / Monitor | 794 | 18.3% | 81.7% |
| live_2026 | Strong | 789 | 9.4% | 90.6% |
