# Rerun Summary

## Route shares
| mode | general | domain | overlap | residual | noise |
| --- | ---: | ---: | ---: | ---: | ---: |
| before (old current) | 3.6% | 3.1% | 20.2% | 70.0% | 3.2% |
| cleaned current | 3.8% | 2.7% | 19.3% | 71.0% | 3.2% |
| cleaned domain_priority | 3.8% | 9.5% | 12.5% | 71.0% | 3.2% |

## Residual quality
- before sample: `62 useful / 56 unclear / 2 noise`
- cleaned current sample: `66 useful / 54 unclear / 0 noise`
- cleaned domain_priority sample: `66 useful / 54 unclear / 0 noise`

## Before / after
- overlap: `20.2% -> 12.5%` (`-7.7 p.p.`)
- residual: `70.0% -> 71.0%` (`+1.0 p.p.`)
- general: `3.6% -> 3.8%`
- domain: `3.1% -> 9.5%`
- noise: `3.2% -> 3.2%`

## Interpretation
- Narrow dictionary cleanup alone helped only a little: overlap `20.2% -> 19.3%`.
- Main effect came from domain-priority conflict resolution on repeated overlap pairs.
- Residual did not shrink, but it became cleaner on the quick sample and is no longer blocked by large unresolved overlap mass.

## Recommendation
- better than before: yes
- final step outcome: overlap moved into target corridor and residual still looks usable
- next move: `go_hdbscan`
