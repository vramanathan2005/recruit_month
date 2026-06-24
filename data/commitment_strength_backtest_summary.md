# Commitment Strength Backtest

- Commitment events tested: 16735
- Known outcomes: 12706
- Decommit/flip outcomes: 3079
- Overall decommit/flip rate: 24.2%

| Outcome | Count |
| --- | ---: |
| stuck | 9627 |
| open | 4029 |
| decommitted | 2837 |
| flipped | 242 |

| Score Bucket | Known Events | Decommit/Flip | Decommit/Flip Rate | Stick Rate |
| --- | ---: | ---: | ---: | ---: |
| Weak / Attack | 820 | 451 | 55.0% | 45.0% |
| Vulnerable | 5432 | 1814 | 33.4% | 66.6% |
| Stable / Monitor | 3410 | 598 | 17.5% | 82.5% |
| Strong | 3044 | 216 | 7.1% | 92.9% |

This backtest scores each historical commitment event using pre-outcome signals only:
timing, prior commitment/decommit history, pre-commit offers/visits, rating tier, distance, and class crowding.
It does not use post-commit events as an input, because those are the outcome being tested.
