# Commitment Strength Holdout Validation

- Known commitment-event outcomes: 14342
- Train sample: classes 2022-2024
- Holdout sample: class 2025
- Live/current-ish sample: class 2026, interpret carefully because outcomes can still be evolving

| Sample | Known Events | Decommit/Flip Rate | AUC |
| --- | ---: | ---: | ---: |
| train_2022_2024 | 7663 | 20.8% | 0.7171 |
| holdout_2025 | 3207 | 20.2% | 0.6978 |
| live_2026 | 3378 | 23.7% | 0.6838 |

| Sample | Bucket | Known Events | Decommit/Flip Rate | Stick Rate |
| --- | --- | ---: | ---: | ---: |
| train_2022_2024 | Weak / Attack | 371 | 48.0% | 52.0% |
| train_2022_2024 | Vulnerable | 2921 | 32.0% | 68.0% |
| train_2022_2024 | Stable / Monitor | 2174 | 15.5% | 84.5% |
| train_2022_2024 | Strong | 2197 | 6.6% | 93.5% |
| holdout_2025 | Weak / Attack | 197 | 47.2% | 52.8% |
| holdout_2025 | Vulnerable | 1333 | 27.3% | 72.7% |
| holdout_2025 | Stable / Monitor | 890 | 16.9% | 83.2% |
| holdout_2025 | Strong | 787 | 5.2% | 94.8% |
| live_2026 | Weak / Attack | 220 | 47.3% | 52.7% |
| live_2026 | Vulnerable | 1535 | 31.5% | 68.5% |
| live_2026 | Stable / Monitor | 898 | 17.4% | 82.6% |
| live_2026 | Strong | 725 | 7.9% | 92.1% |
