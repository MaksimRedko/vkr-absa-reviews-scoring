# phase4_anchor_residual_routing_diagnostic

## Route counts
- general: 900 / 24857 (3.6%)
- domain: 764 / 24857 (3.1%)
- overlap: 5013 / 24857 (20.2%)
- residual: 17394 / 24857 (70.0%)
- noise: 786 / 24857 (3.2%)

## Conflict
- general vs domain conflict: strong
- overlap rows: 5013 (20.2%)

## Residual sample
- quick labels on sample: useful=62, noise=2, unclear=56
- residual judgement: useful material for HDBSCAN

## HDBSCAN readiness
- residual raw size: 17394
- residual cleaned size: 10061
- diagnostic clustering ran: true
- diagnostic clusters: 179
- clustered share: 0.6621

## Recommendation
- do_not_go_hdbscan
