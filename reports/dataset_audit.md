# Dataset audit report

## Source
- Dataset: Experimental Surface Roughness Dataset for CNC End Milling (straight segments)
- DOI: 10.5281/zenodo.21518387 (CC BY 4.0)
- Material: St3ps carbon structural steel; 12 mm 4-flute coated end mill
- Operation: finish end milling; programmed feed/spindle from CNC (not independently measured)

## Counts
- Independent machined segments: **15**
- Surface profilograms (profiles): **45**
- Machining runs: **5**

## Labels and inputs
- Primary target: `ra_gaussian_filtered_um` (Gaussian L-filter, λc=0.8 mm, central 4 mm evaluation)
- Predictors: programmed feed/spindle bands, tool/workpiece constants, geometry/run categoricals
- **No in-process sensor signals** are included in this public release.

## Quality checks
- Missing target values: 0
- Duplicate profile keys: 0
- Ra range (µm): 0.175 – 3.396

## Validity considerations
- Three repeated profilograms per segment share the same programmed regime → use grouped CV by `segment_id`.
- Variable feed/spindle within a segment is summarized by min/max/mean; true instantaneous values are not logged.
- Geometry (scan axis vs feed) affects interpretability of texture; do not over-interpret spectral diagnostics.
- Small segment count (15) limits hyperparameter tuning and external generalisation claims.
