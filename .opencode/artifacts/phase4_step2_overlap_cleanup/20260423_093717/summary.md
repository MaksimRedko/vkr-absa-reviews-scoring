# phase4_anchor_residual_routing_diagnostic

## Mode
- routing mode: `current`

## Route counts
- general: 955 / 24857 (3.8%)
- domain: 675 / 24857 (2.7%)
- overlap: 4796 / 24857 (19.3%)
- residual: 17644 / 24857 (71.0%)
- noise: 787 / 24857 (3.2%)

## Conflict
- general vs domain conflict: strong
- overlap rows: 4796 (19.3%)

## Residual sample
- quick labels on sample: useful=66, noise=0, unclear=54
- residual judgement: useful material for HDBSCAN

## HDBSCAN readiness
- residual raw size: 17644
- residual cleaned size: 10252
- diagnostic clustering ran: true
- diagnostic clusters: 177
- clustered share: 0.6478

## Recommendation
- do_not_go_hdbscan
