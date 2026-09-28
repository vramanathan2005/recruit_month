# Flip snapshot model

- Data cutoff: 2026-09-26. Snapshots every 14 days of every active commitment; target = decommit/flip within 60 days.
- Labeled snapshots: 181,500 (6.7% break within 60 days). Each test class (2025, 2026, 2027) is scored by models trained only on the classes before it.
- Live board model: boosted (higher test AUC), refit on all labeled snapshots; 2,018 active commitments scored as of 2026-09-26.

## Test classes

| model | rows | sample | snapshots | breaks_within_60_days | actual_rate | average_predicted | auc | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.3044 | 0.5992 | 0.1514 | 0.48 |
| logistic | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.0589 | 0.6553 | 0.0767 | 0.2849 |
| boosted | same as at-commit | class_2025 | 31703 | 2712 | 0.0855 | 0.0621 | 0.7086 | 0.0746 | 0.2747 |
| logistic | all snapshots | class_2025 | 35273 | 2712 | 0.0769 | 0.0578 | 0.6633 | 0.0693 | 0.2612 |
| boosted | all snapshots | class_2025 | 35273 | 2712 | 0.0769 | 0.0596 | 0.7215 | 0.0674 | 0.2508 |
| at_commit | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.3095 | 0.6035 | 0.1549 | 0.4781 |
| logistic | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.0666 | 0.6496 | 0.0818 | 0.2992 |
| boosted | same as at-commit | class_2026 | 33783 | 3114 | 0.0922 | 0.0656 | 0.7394 | 0.0782 | 0.2795 |
| logistic | all snapshots | class_2026 | 35014 | 3114 | 0.0889 | 0.0672 | 0.6428 | 0.0793 | 0.2918 |
| boosted | all snapshots | class_2026 | 35014 | 3114 | 0.0889 | 0.0651 | 0.7408 | 0.0756 | 0.2716 |
| at_commit | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.6034 | 0.3209 | 0.399 | 1.072 |
| logistic | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.0792 | 0.6115 | 0.3253 | 1.0421 |
| boosted | same as at-commit | class_2027 | 1695 | 656 | 0.387 | 0.0764 | 0.6122 | 0.3263 | 1.0798 |
| logistic | all snapshots | class_2027 | 12993 | 656 | 0.0505 | 0.0609 | 0.6891 | 0.0463 | 0.1886 |
| boosted | all snapshots | class_2027 | 12993 | 656 | 0.0505 | 0.0395 | 0.7983 | 0.0444 | 0.1713 |
| logistic | training data | train_2022_2024 | 97879 | 5688 | 0.0581 | 0.0577 | 0.6948 | 0.0528 | 0.2077 |
| boosted | training data | train_2022_2024 | 97879 | 5688 | 0.0581 | 0.0552 | 0.8952 | 0.046 | 0.1669 |

## Weekly board check (top 25 on each date)

| model | rows | class_year | board_dates | top_25_hit_rate | board_base_rate | lift |
| --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | 2025 | 35 | 0.2046 | 0.0892 | 2.29 |
| logistic | same as at-commit | 2025 | 35 | 0.248 | 0.0892 | 2.78 |
| boosted | same as at-commit | 2025 | 35 | 0.2789 | 0.0892 | 3.13 |
| logistic | all snapshots | 2025 | 35 | 0.2389 | 0.0783 | 3.05 |
| boosted | all snapshots | 2025 | 35 | 0.2629 | 0.0783 | 3.36 |
| at_commit | same as at-commit | 2026 | 36 | 0.2278 | 0.089 | 2.56 |
| logistic | same as at-commit | 2026 | 36 | 0.2489 | 0.089 | 2.8 |
| boosted | same as at-commit | 2026 | 36 | 0.3089 | 0.089 | 3.47 |
| logistic | all snapshots | 2026 | 36 | 0.23 | 0.0825 | 2.79 |
| boosted | all snapshots | 2026 | 36 | 0.2778 | 0.0825 | 3.37 |
| at_commit | same as at-commit | 2027 | 0 |  |  |  |
| logistic | same as at-commit | 2027 | 0 |  |  |  |
| boosted | same as at-commit | 2027 | 0 |  |  |  |
| logistic | all snapshots | 2027 | 21 | 0.2057 | 0.0834 | 2.47 |
| boosted | all snapshots | 2027 | 21 | 0.2762 | 0.0834 | 3.31 |

## Calibration (test classes, raw model)

| model | bucket | snapshots | predicted | actual |
| --- | --- | --- | --- | --- |
| logistic | 0-2% | 2949 | 0.016 | 0.0203 |
| logistic | 2-5% | 34303 | 0.0367 | 0.0466 |
| logistic | 5-10% | 37133 | 0.0686 | 0.086 |
| logistic | 10-20% | 7692 | 0.1295 | 0.1668 |
| logistic | 20-35% | 998 | 0.2455 | 0.2705 |
| logistic | 35%+ | 205 | 0.4459 | 0.3659 |
| boosted | 0-2% | 16323 | 0.0123 | 0.0175 |
| boosted | 2-5% | 31209 | 0.0335 | 0.0406 |
| boosted | 5-10% | 21975 | 0.0719 | 0.0939 |
| boosted | 10-20% | 12043 | 0.1323 | 0.186 |
| boosted | 20-35% | 1498 | 0.2455 | 0.3198 |
| boosted | 35%+ | 232 | 0.4752 | 0.625 |

## Recalibration by stage of the cycle (fit on one class, checked on the other)

The live board's `break_probability_60d` is the raw score recalibrated this way, fit on both classes.

| fit_on | checked_on | stage | snapshots | raw | calibrated | actual |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | 2026 | 180+ days out | 7674 | 0.0637 | 0.0936 | 0.0722 |
| 2025 | 2026 | 90-180 days out | 12265 | 0.0379 | 0.0278 | 0.0396 |
| 2025 | 2026 | 60-90 days out | 4304 | 0.0828 | 0.129 | 0.1364 |
| 2025 | 2026 | 30-60 days out | 4332 | 0.1108 | 0.1387 | 0.1674 |
| 2025 | 2026 | 0-30 days out | 4857 | 0.0911 | 0.108 | 0.1295 |
| 2025 | 2026 | after early signing day | 1582 | 0.0304 | 0.1044 | 0.0841 |
| 2026 | 2025 | 180+ days out | 6720 | 0.0507 | 0.057 | 0.0747 |
| 2026 | 2025 | 90-180 days out | 11459 | 0.0393 | 0.0411 | 0.0291 |
| 2026 | 2025 | 60-90 days out | 4307 | 0.0721 | 0.1192 | 0.1131 |
| 2026 | 2025 | 30-60 days out | 4380 | 0.1074 | 0.1622 | 0.1342 |
| 2026 | 2025 | 0-30 days out | 5288 | 0.0882 | 0.1255 | 0.105 |
| 2026 | 2025 | after early signing day | 3119 | 0.0203 | 0.0513 | 0.0792 |

## Before signing: breaks at any point before the signing period ends

Trained on complete classes only; each checked on a model trained on earlier classes and recalibrated on the other. The live board's `break_before_signing` uses all of them. Ranges (`*_low`/`*_high`) are the 10th-90th percentile of 6 models trained on resampled commitments.

| checked_on | stage | snapshots | auc | calibrated | actual | top_25_hit_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2026 | all | 38325 | 0.7065 | 0.232 | 0.2395 | 0.572 |
| 2026 | 90-180 days out | 12350 | 0.6259 | 0.2285 | 0.2504 |  |
| 2026 | 60-90 days out | 4375 | 0.6391 | 0.219 | 0.2347 |  |
| 2026 | 30-60 days out | 4433 | 0.665 | 0.2051 | 0.2093 |  |
| 2026 | 0-30 days out | 7036 | 0.7284 | 0.1141 | 0.1308 |  |
| 2026 | 180+ days out | 7772 | 0.678 | 0.4034 | 0.3865 |  |
| 2026 | after early signing day | 2359 | 0.8722 | 0.1126 | 0.0877 |  |
| 2025 | all | 38734 | 0.7317 | 0.2115 | 0.2043 | 0.5621 |
| 2025 | 180+ days out | 6805 | 0.7344 | 0.3574 | 0.3669 |  |
| 2025 | 90-180 days out | 11527 | 0.6717 | 0.2386 | 0.2167 |  |
| 2025 | 60-90 days out | 4348 | 0.6764 | 0.2206 | 0.204 |  |
| 2025 | 30-60 days out | 4487 | 0.676 | 0.1956 | 0.1917 |  |
| 2025 | 0-30 days out | 7192 | 0.6971 | 0.132 | 0.1146 |  |
| 2025 | after early signing day | 4375 | 0.8028 | 0.0512 | 0.0795 |  |

