# ComptonNet
These notebooks present preliminary experiments on GRB localization using an unbinned analysis of raw features from the Compton data space—energy, φ (phi), χ (chi), ψ (psi), and distance. Performance is reported via angular deviation (mean and median). We evaluate Deep Sets (well-suited for permutation-invariant event sets) and Set Transformers. In parallel, we are exploring a binned approach that constructs 3D histograms over the angular features (φ, χ, ψ) and trains 3D CNNs for localization. Our goal is an ML-based GRB localization tool that is faster than current TS-map pipelines, enabling quicker, more precise follow-up by optical and infrared surveys.


# Results

Using Toy Data with Resolution = 5 deg
No of Compton Events | No of Background Events | Mean Angular Deviation | Median Angular Deviation | RMS Angular Deviation 
--- | --- | --- | --- | --- 
2000 | 0 | 93.91 | | 
1333 | 667 | 93.91 | | 
1000 | 1000 | 93.91 | | 


Using Toy Data with Resolution = 3 deg
No of Compton Events | No of Background Events | Mean Angular Deviation | Median Angular Deviation | RMS Angular Deviation 
--- | --- | --- | --- | --- 
2000 | 0 | 93.91 | | 
1333 | 667 | 93.91 | | 
1000 | 1000 | 93.91 | | 
