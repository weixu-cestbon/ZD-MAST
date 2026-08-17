# Feature contract

- m/z window: [2000, 20000) Da
- bin width: 3 Da
- features: 6000
- within-bin aggregation: maximum
- intensity transform: square root
- spectrum-level L2 normalization
- sample-level replicate aggregation: arithmetic mean followed by L2 normalization
- ZD-MAST-A peak_presence6000: nonzero bins from sparse peak-list-like mzML exports
- ZD-MAST-B peak_presence6000: frozen label-free peak extraction from converted dense profiles (Savitzky-Golay window 15, polynomial order 3; Gaussian baseline sigma 150 points; prominence 3 x robust residual sigma; minimum distance 5 points), followed by 3-Da bin presence and sample-level OR aggregation

Site A open tables preserve vendor-processed mzML export peak-list-like values. Site B open tables preserve converted dense profile values. Equal dimensions do not imply identical spectral semantics.
