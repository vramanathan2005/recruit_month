# Flip snapshot model

- Data cutoff: 2026-09-26. Snapshots every 14 days of every active commitment; target = decommit/flip within 60 days.
- Labeled snapshots: 181,183 (6.7% break within 60 days). Each test class (2024, 2025, 2026, 2027) is scored by models trained only on the classes before it.
- Live board model: boosted (higher test AUC), refit on all labeled snapshots; 2,008 active commitments scored as of 2026-09-26.

## Test classes

| model | rows | sample | snapshots | breaks_within_60_days | actual_rate | average_predicted | auc | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | class_2024 | 26023 | 2045 | 0.0786 | 0.2824 | 0.656 | 0.136 | 0.4353 |
| logistic | same as at-commit | class_2024 | 26023 | 2045 | 0.0786 | 0.0571 | 0.6793 | 0.0709 | 0.2654 |
| boosted | same as at-commit | class_2024 | 26023 | 2045 | 0.0786 | 0.0604 | 0.7139 | 0.0692 | 0.2568 |
| logistic | all snapshots | class_2024 | 37685 | 2045 | 0.0543 | 0.0574 | 0.6626 | 0.0502 | 0.2018 |
| boosted | all snapshots | class_2024 | 37685 | 2045 | 0.0543 | 0.0564 | 0.7354 | 0.0492 | 0.1927 |
| at_commit | same as at-commit | class_2025 | 31556 | 2638 | 0.0836 | 0.3044 | 0.6024 | 0.1506 | 0.4779 |
| logistic | same as at-commit | class_2025 | 31556 | 2638 | 0.0836 | 0.0589 | 0.6594 | 0.0749 | 0.2789 |
| boosted | same as at-commit | class_2025 | 31556 | 2638 | 0.0836 | 0.0598 | 0.7193 | 0.0728 | 0.2677 |
| logistic | all snapshots | class_2025 | 35115 | 2638 | 0.0751 | 0.058 | 0.6653 | 0.0676 | 0.2559 |
| boosted | all snapshots | class_2025 | 35115 | 2638 | 0.0751 | 0.0572 | 0.7325 | 0.0656 | 0.2442 |
| at_commit | same as at-commit | class_2026 | 33712 | 3066 | 0.0909 | 0.3091 | 0.6046 | 0.1541 | 0.4763 |
| logistic | same as at-commit | class_2026 | 33712 | 3066 | 0.0909 | 0.0693 | 0.647 | 0.0804 | 0.2951 |
| boosted | same as at-commit | class_2026 | 33712 | 3066 | 0.0909 | 0.0679 | 0.7498 | 0.0763 | 0.2733 |
| logistic | all snapshots | class_2026 | 34943 | 3066 | 0.0877 | 0.0699 | 0.64 | 0.0779 | 0.2879 |
| boosted | all snapshots | class_2026 | 34943 | 3066 | 0.0877 | 0.0674 | 0.7514 | 0.0737 | 0.2656 |
| at_commit | same as at-commit | class_2027 | 1693 | 655 | 0.3869 | 0.6035 | 0.3208 | 0.3992 | 1.0725 |
| logistic | same as at-commit | class_2027 | 1693 | 655 | 0.3869 | 0.0878 | 0.6137 | 0.3182 | 1.0139 |
| boosted | same as at-commit | class_2027 | 1693 | 655 | 0.3869 | 0.112 | 0.612 | 0.3021 | 0.948 |
| logistic | all snapshots | class_2027 | 12988 | 655 | 0.0504 | 0.0637 | 0.7067 | 0.0465 | 0.1878 |
| boosted | all snapshots | class_2027 | 12988 | 655 | 0.0504 | 0.0556 | 0.8156 | 0.0429 | 0.1667 |
| logistic | training data | train_2022_2023 | 60111 | 3612 | 0.0601 | 0.059 | 0.7078 | 0.0541 | 0.2115 |
| boosted | training data | train_2022_2023 | 60111 | 3612 | 0.0601 | 0.0702 | 0.8979 | 0.0446 | 0.167 |

## Weekly board check (top 25 on each date)

| model | rows | class_year | board_dates | top_25_hit_rate | board_base_rate | lift |
| --- | --- | --- | --- | --- | --- | --- |
| at_commit | same as at-commit | 2024 | 30 | 0.2107 | 0.0757 | 2.78 |
| logistic | same as at-commit | 2024 | 30 | 0.2173 | 0.0757 | 2.87 |
| boosted | same as at-commit | 2024 | 30 | 0.2307 | 0.0757 | 3.05 |
| logistic | all snapshots | 2024 | 31 | 0.2013 | 0.0574 | 3.5 |
| boosted | all snapshots | 2024 | 31 | 0.1935 | 0.0574 | 3.37 |
| at_commit | same as at-commit | 2025 | 35 | 0.2046 | 0.0872 | 2.35 |
| logistic | same as at-commit | 2025 | 35 | 0.2571 | 0.0872 | 2.95 |
| boosted | same as at-commit | 2025 | 35 | 0.312 | 0.0872 | 3.58 |
| logistic | all snapshots | 2025 | 35 | 0.2469 | 0.0766 | 3.22 |
| boosted | all snapshots | 2025 | 35 | 0.2914 | 0.0766 | 3.8 |
| at_commit | same as at-commit | 2026 | 36 | 0.2333 | 0.0895 | 2.61 |
| logistic | same as at-commit | 2026 | 36 | 0.2644 | 0.0895 | 2.95 |
| boosted | same as at-commit | 2026 | 36 | 0.3511 | 0.0895 | 3.92 |
| logistic | all snapshots | 2026 | 36 | 0.2489 | 0.0828 | 3.01 |
| boosted | all snapshots | 2026 | 36 | 0.3278 | 0.0828 | 3.96 |
| at_commit | same as at-commit | 2027 | 0 |  |  |  |
| logistic | same as at-commit | 2027 | 0 |  |  |  |
| boosted | same as at-commit | 2027 | 0 |  |  |  |
| logistic | all snapshots | 2027 | 21 | 0.2152 | 0.0835 | 2.58 |
| boosted | all snapshots | 2027 | 21 | 0.2686 | 0.0835 | 3.22 |

## Calibration (test classes, raw model)

| model | bucket | snapshots | predicted | actual |
| --- | --- | --- | --- | --- |
| logistic | 0-2% | 3711 | 0.0161 | 0.0162 |
| logistic | 2-5% | 52588 | 0.0368 | 0.0424 |
| logistic | 5-10% | 52385 | 0.068 | 0.0755 |
| logistic | 10-20% | 10167 | 0.1293 | 0.1589 |
| logistic | 20-35% | 1392 | 0.2533 | 0.2579 |
| logistic | 35%+ | 488 | 0.4988 | 0.375 |
| boosted | 0-2% | 18691 | 0.0119 | 0.0147 |
| boosted | 2-5% | 50232 | 0.0335 | 0.0342 |
| boosted | 5-10% | 32783 | 0.0701 | 0.0814 |
| boosted | 10-20% | 15909 | 0.1324 | 0.1716 |
| boosted | 20-35% | 2623 | 0.2523 | 0.2856 |
| boosted | 35%+ | 493 | 0.4797 | 0.5335 |

## Recalibration by stage of the cycle (fit on one class, checked on the other)

The live board's `break_probability_60d` is the raw score recalibrated this way, fit on both classes.

| fit_on | checked_on | stage | snapshots | raw | calibrated | actual |
| --- | --- | --- | --- | --- | --- | --- |
| 2025 | 2026 | 180+ days out | 7662 | 0.0663 | 0.1052 | 0.0728 |
| 2025 | 2026 | 90-180 days out | 12252 | 0.0376 | 0.0287 | 0.04 |
| 2025 | 2026 | 60-90 days out | 4298 | 0.0875 | 0.1424 | 0.1342 |
| 2025 | 2026 | 30-60 days out | 4326 | 0.1171 | 0.1495 | 0.1625 |
| 2025 | 2026 | 0-30 days out | 4844 | 0.0944 | 0.1081 | 0.1249 |
| 2025 | 2026 | after early signing day | 1561 | 0.0288 | 0.1074 | 0.0852 |
| 2026 | 2025 | 180+ days out | 6720 | 0.0478 | 0.051 | 0.0747 |
| 2026 | 2025 | 90-180 days out | 11459 | 0.038 | 0.0406 | 0.0291 |
| 2026 | 2025 | 60-90 days out | 4307 | 0.0688 | 0.1069 | 0.1131 |
| 2026 | 2025 | 30-60 days out | 4380 | 0.104 | 0.1457 | 0.1342 |
| 2026 | 2025 | 0-30 days out | 5249 | 0.0854 | 0.1142 | 0.0983 |
| 2026 | 2025 | after early signing day | 3000 | 0.0178 | 0.049 | 0.0707 |

## Coverage blend (fit on one held-out class, checked on the other)

next 60 days: blend not used (no improvement); before signing: blend not used (no improvement)

| target | blend | fit_on | checked_on | auc | top_25_hit_rate | predicted | actual |
| --- | --- | --- | --- | --- | --- | --- | --- |
| next 60 days | False | 2025 | 2026 | 0.751 | 0.3278 | 0.0889 | 0.0877 |
| next 60 days | False | 2026 | 2025 | 0.7457 | 0.2914 | 0.0756 | 0.0751 |
| next 60 days | True | 2025 | 2026 | 0.7494 | 0.3233 | 0.09 | 0.0877 |
| next 60 days | True | 2026 | 2025 | 0.7461 | 0.2994 | 0.0752 | 0.0751 |
| before signing | False | 2025 | 2026 | 0.7185 | 0.607 | 0.2334 | 0.2354 |
| before signing | False | 2026 | 2025 | 0.7486 | 0.5884 | 0.2043 | 0.1982 |
| before signing | True | 2025 | 2026 | 0.7165 | 0.572 | 0.2356 | 0.2354 |
| before signing | True | 2026 | 2025 | 0.749 | 0.6074 | 0.2042 | 0.1982 |

## Before signing: breaks at any point before the signing period ends

Trained on complete classes only; each checked on a model trained on earlier classes and recalibrated on the other. The live board's `break_before_signing` uses all of them. Ranges (`*_low`/`*_high`) are the 10th-90th percentile of 6 models trained on resampled commitments.

| checked_on | stage | snapshots | auc | calibrated | actual | top_25_hit_rate |
| --- | --- | --- | --- | --- | --- | --- |
| 2026 | all | 38251 | 0.7111 | 0.2334 | 0.2354 | 0.607 |
| 2026 | 90-180 days out | 12338 | 0.625 | 0.225 | 0.2458 |  |
| 2026 | 60-90 days out | 4369 | 0.6486 | 0.2156 | 0.23 |  |
| 2026 | 30-60 days out | 4428 | 0.6754 | 0.2029 | 0.2048 |  |
| 2026 | 0-30 days out | 7023 | 0.7375 | 0.1066 | 0.1257 |  |
| 2026 | 180+ days out | 7761 | 0.6903 | 0.4299 | 0.3826 |  |
| 2026 | after early signing day | 2332 | 0.8827 | 0.0973 | 0.0888 |  |
| 2025 | all | 38572 | 0.7368 | 0.2043 | 0.1982 | 0.5884 |
| 2025 | 180+ days out | 6805 | 0.7365 | 0.3344 | 0.3641 |  |
| 2025 | 90-180 days out | 11527 | 0.6733 | 0.2325 | 0.2092 |  |
| 2025 | 60-90 days out | 4348 | 0.6805 | 0.2136 | 0.1971 |  |
| 2025 | 30-60 days out | 4487 | 0.6833 | 0.1877 | 0.1845 |  |
| 2025 | 0-30 days out | 7189 | 0.7067 | 0.1279 | 0.1078 |  |
| 2025 | after early signing day | 4216 | 0.818 | 0.0557 | 0.07 |  |

