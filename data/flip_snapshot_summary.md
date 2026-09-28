# Flip snapshot model

- Data cutoff: 2026-05-31. Snapshots every 14 days of every active commitment; target = decommit/flip within 60 days.
- Labeled snapshots: 172,402 (6.9% break within 60 days). Each test class (2025, 2026) is scored by models trained only on the classes before it.
- Live board model: boosted (higher test AUC), refit on all labeled snapshots; 613 active commitments scored as of 2026-05-31.

## Test classes

| model | rows | sample | snapshots | breaks_within_60_days | actual_rate | average_predicted | auc | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.3043 | 0.5993 | 0.1513 | 0.4796 |
| logistic | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.0586 | 0.6556 | 0.077 | 0.2855 |
| boosted | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.061 | 0.7025 | 0.075 | 0.2763 |
| logistic | all snapshots | class_2025 | 35273 | 2712 | 0.0769 | 0.0576 | 0.6637 | 0.0695 | 0.2618 |
| boosted | all snapshots | class_2025 | 35273 | 2712 | 0.0769 | 0.0584 | 0.7166 | 0.0677 | 0.252 |
| at_commit | same as at-commit | class_2026 | 34510 | 3233 | 0.0937 | 0.3154 | 0.5929 | 0.1598 | 0.4906 |
| logistic | same as at-commit | class_2026 | 34510 | 3233 | 0.0937 | 0.066 | 0.6353 | 0.0835 | 0.3056 |
| boosted | same as at-commit | class_2026 | 34510 | 3233 | 0.0937 | 0.0651 | 0.7213 | 0.0803 | 0.2881 |
| logistic | all snapshots | class_2026 | 36383 | 3233 | 0.0889 | 0.0666 | 0.6275 | 0.0796 | 0.294 |
| boosted | all snapshots | class_2026 | 36383 | 3233 | 0.0889 | 0.064 | 0.7263 | 0.0763 | 0.2756 |
| logistic | training data | train_2022_2024 | 97895 | 5688 | 0.0581 | 0.0577 | 0.683 | 0.0531 | 0.2096 |
| boosted | training data | train_2022_2024 | 97895 | 5688 | 0.0581 | 0.0546 | 0.8913 | 0.047 | 0.1701 |

## Weekly board check (top 25 on each date)

| model | rows | class_year | board_dates | top_25_hit_rate | board_base_rate | lift |
| --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | 2025 | 35 | 0.2069 | 0.0892 | 2.32 |
| logistic | same as at-commit | 2025 | 35 | 0.2583 | 0.0892 | 2.89 |
| boosted | same as at-commit | 2025 | 35 | 0.28 | 0.0892 | 3.14 |
| logistic | all snapshots | 2025 | 35 | 0.2411 | 0.0783 | 3.08 |
| boosted | all snapshots | 2025 | 35 | 0.2697 | 0.0783 | 3.45 |
| at_commit | same as at-commit | 2026 | 35 | 0.2366 | 0.0976 | 2.42 |
| logistic | same as at-commit | 2026 | 35 | 0.2469 | 0.0976 | 2.53 |
| boosted | same as at-commit | 2026 | 35 | 0.3097 | 0.0976 | 3.17 |
| logistic | all snapshots | 2026 | 36 | 0.2022 | 0.0819 | 2.47 |
| boosted | all snapshots | 2026 | 36 | 0.2711 | 0.0819 | 3.31 |

## Calibration (test classes)

| model | bucket | snapshots | predicted | actual |
| --- | --- | --- | --- | --- |
| logistic | 0-2% | 2215 | 0.0161 | 0.0158 |
| logistic | 2-5% | 28619 | 0.037 | 0.0524 |
| logistic | 5-10% | 33622 | 0.0687 | 0.0916 |
| logistic | 10-20% | 6365 | 0.127 | 0.1716 |
| logistic | 20-35% | 678 | 0.2511 | 0.264 |
| logistic | 35%+ | 157 | 0.4523 | 0.3822 |
| boosted | 0-2% | 12957 | 0.0126 | 0.0203 |
| boosted | 2-5% | 25006 | 0.0335 | 0.0457 |
| boosted | 5-10% | 20033 | 0.0727 | 0.0972 |
| boosted | 10-20% | 12705 | 0.1298 | 0.1721 |
| boosted | 20-35% | 763 | 0.247 | 0.3617 |
| boosted | 35%+ | 192 | 0.4759 | 0.6667 |

