# Commitment Strength Backtest

- Commitment events tested: 16788
- Known outcomes: 12721
- Decommit/flip outcomes: 3092
- Overall decommit/flip rate: 24.3%

| Outcome | Count |
| --- | ---: |
| stuck | 9629 |
| open | 4067 |
| decommitted | 2849 |
| flipped | 243 |

| Score Bucket | Known Events | Decommit/Flip | Decommit/Flip Rate | Stick Rate |
| --- | ---: | ---: | ---: | ---: |
| Weak / Attack | 829 | 460 | 55.5% | 44.5% |
| Vulnerable | 5431 | 1816 | 33.4% | 66.6% |
| Stable / Monitor | 3415 | 599 | 17.5% | 82.5% |
| Strong | 3046 | 217 | 7.1% | 92.9% |

This backtest scores each historical commitment event using pre-outcome signals only:
timing, prior commitment/decommit history, pre-commit offers/visits, rating tier, distance, and class crowding.
It does not use post-commit events as an input, because those are the outcome being tested.
