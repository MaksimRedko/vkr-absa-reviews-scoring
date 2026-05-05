go_hdbscan

Overlap dropped from `20.2%` to `12.5%` after targeted cleanup and domain-priority conflict routing.
Residual stayed large (`71.0%`) but the quick sample remained usable (`66 useful / 54 unclear / 0 noise`).
The blocking issue is no longer the two-layer collision itself.
There is now enough signal quality to run HDBSCAN on residual as the next isolated step.
