# phase4_anchor_residual_routing_diagnostic

## Route counts
- general: 1090 / 24857 (4.4%)
- domain: 291 / 24857 (1.2%)
- overlap: 5336 / 24857 (21.5%)
- residual: 17110 / 24857 (68.8%)
- noise: 1030 / 24857 (4.1%)

## Conflict
- general vs domain conflict: strong
- overlap rows: 5336 (21.5%)

## Residual sample
- quick labels on sample: useful=39, noise=1, unclear=80
- residual judgement: mostly trash / too unclear

## HDBSCAN readiness
- residual raw size: 17110
- residual cleaned size: 9368
- diagnostic clustering ran: true
- diagnostic clusters: 170
- clustered share: 0.6648

## Recommendation
- do_not_go_hdbscan
