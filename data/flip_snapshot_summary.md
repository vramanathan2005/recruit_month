# Flip snapshot model

- Data cutoff: 2026-09-26. Snapshots every 14 days of every active commitment; target = decommit/flip within 60 days.
- Labeled snapshots: 181,500 (6.7% break within 60 days). Each test class (2025, 2026, 2027) is scored by models trained only on the classes before it.
- Live board model: boosted (higher test AUC), refit on all labeled snapshots; 2,018 active commitments scored as of 2026-09-26.

## Test classes

| model | rows | sample | snapshots | breaks_within_60_days | actual_rate | average_predicted | auc | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.3044 | 0.5992 | 0.1514 | 0.48 |
| logistic | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.0586 | 0.6557 | 0.0769 | 0.2855 |
| boosted | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.0616 | 0.704 | 0.075 | 0.2761 |
| logistic | all snapshots | class_2025 | 35273 | 2712 | 0.0769 | 0.0576 | 0.6638 | 0.0695 | 0.2617 |
| boosted | all snapshots | class_2025 | 35273 | 2712 | 0.0769 | 0.0591 | 0.7176 | 0.0676 | 0.2519 |
| at_commit | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.3095 | 0.6035 | 0.1549 | 0.4781 |
| logistic | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.0671 | 0.6413 | 0.0821 | 0.3005 |
| boosted | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.0648 | 0.7307 | 0.079 | 0.2826 |
| logistic | all snapshots | class_2026 | 35014 | 3114 | 0.0889 | 0.0678 | 0.6336 | 0.0795 | 0.2931 |
| boosted | all snapshots | class_2026 | 35014 | 3114 | 0.0889 | 0.0644 | 0.7322 | 0.0763 | 0.2746 |
| at_commit | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.6034 | 0.3209 | 0.399 | 1.072 |
| logistic | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.0768 | 0.6128 | 0.3278 | 1.0518 |
| boosted | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.0745 | 0.6056 | 0.3281 | 1.0865 |
| logistic | all snapshots | class_2027 | 12993 | 656 | 0.0505 | 0.0608 | 0.6821 | 0.0467 | 0.1901 |
| boosted | all snapshots | class_2027 | 12993 | 656 | 0.0505 | 0.0392 | 0.7973 | 0.0446 | 0.1722 |
| logistic | training data | train_2022_2024 | 97879 | 5688 | 0.0581 | 0.0578 | 0.683 | 0.0531 | 0.2096 |
| boosted | training data | train_2022_2024 | 97879 | 5688 | 0.0581 | 0.0549 | 0.8924 | 0.0469 | 0.17 |

## Weekly board check (top 25 on each date)

| model | rows | class_year | board_dates | top_25_hit_rate | board_base_rate | lift |
| --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | 2025 | 35 | 0.2046 | 0.0892 | 2.29 |
| logistic | same as at-commit | 2025 | 35 | 0.2594 | 0.0892 | 2.91 |
| boosted | same as at-commit | 2025 | 35 | 0.2834 | 0.0892 | 3.18 |
| logistic | all snapshots | 2025 | 35 | 0.2423 | 0.0783 | 3.1 |
| boosted | all snapshots | 2025 | 35 | 0.2617 | 0.0783 | 3.34 |
| at_commit | same as at-commit | 2026 | 36 | 0.2278 | 0.089 | 2.56 |
| logistic | same as at-commit | 2026 | 36 | 0.2389 | 0.089 | 2.68 |
| boosted | same as at-commit | 2026 | 36 | 0.2989 | 0.089 | 3.36 |
| logistic | all snapshots | 2026 | 36 | 0.2067 | 0.0825 | 2.51 |
| boosted | all snapshots | 2026 | 36 | 0.2644 | 0.0825 | 3.21 |
| at_commit | same as at-commit | 2027 | 0 |  |  |  |
| logistic | same as at-commit | 2027 | 0 |  |  |  |
| boosted | same as at-commit | 2027 | 0 |  |  |  |
| logistic | all snapshots | 2027 | 21 | 0.2038 | 0.0834 | 2.44 |
| boosted | all snapshots | 2027 | 21 | 0.2514 | 0.0834 | 3.01 |

## Calibration (test classes, raw model)

| model | bucket | snapshots | predicted | actual |
| --- | --- | --- | --- | --- |
| logistic | 0-2% | 2405 | 0.0162 | 0.0162 |
| logistic | 2-5% | 33474 | 0.037 | 0.0461 |
| logistic | 5-10% | 38741 | 0.0687 | 0.0874 |
| logistic | 10-20% | 7646 | 0.1271 | 0.1615 |
| logistic | 20-35% | 839 | 0.2479 | 0.267 |
| logistic | 35%+ | 175 | 0.4391 | 0.32 |
| boosted | 0-2% | 16535 | 0.0119 | 0.0185 |
| boosted | 2-5% | 29975 | 0.0336 | 0.0403 |
| boosted | 5-10% | 22267 | 0.0716 | 0.0962 |
| boosted | 10-20% | 13337 | 0.1293 | 0.1779 |
| boosted | 20-35% | 978 | 0.2435 | 0.3466 |
| boosted | 35%+ | 188 | 0.4722 | 0.6011 |

## Recalibration by stage of the cycle (fit on one class, checked on the other)

The live board's `break_probability_60d` is the raw score recalibrated this way, fit on both classes.

| fit_on | checked_on | stage | snapshots | raw | calibrated | actual |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | 2026 | 180+ days out | 7674 | 0.0642 | 0.0945 | 0.0722 |
| 2025 | 2026 | 90-180 days out | 12265 | 0.038 | 0.0281 | 0.0396 |
| 2025 | 2026 | 60-90 days out | 4304 | 0.0819 | 0.1272 | 0.1364 |
| 2025 | 2026 | 30-60 days out | 4332 | 0.1062 | 0.1363 | 0.1674 |
| 2025 | 2026 | 0-30 days out | 4857 | 0.0893 | 0.1073 | 0.1295 |
| 2025 | 2026 | after early signing day | 1582 | 0.0313 | 0.1094 | 0.0841 |
| 2026 | 2025 | 180+ days out | 6720 | 0.0509 | 0.0568 | 0.0747 |
| 2026 | 2025 | 90-180 days out | 11459 | 0.0393 | 0.0408 | 0.0291 |
| 2026 | 2025 | 60-90 days out | 4307 | 0.0722 | 0.1209 | 0.1131 |
| 2026 | 2025 | 30-60 days out | 4380 | 0.1046 | 0.1651 | 0.1342 |
| 2026 | 2025 | 0-30 days out | 5288 | 0.0871 | 0.1267 | 0.105 |
| 2026 | 2025 | after early signing day | 3119 | 0.0199 | 0.0487 | 0.0792 |

