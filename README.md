# COmpton Spectrometer and Imager (COSI) - GRBLocalization

These notebooks present preliminary experiments on GRB localization using an unbinned analysis of raw features from the Compton data space—energy, φ (phi), χ (chi), ψ (psi), and distance. Performance is reported via angular deviation (mean and median). We evaluate Deep Sets (well-suited for permutation-invariant event sets) and Set Transformers. In parallel, we are exploring a binned approach that constructs 3D histograms over the angular features (φ, χ, ψ) and trains 3D CNNs for localization. Our goal is an ML-based GRB localization tool that is faster than current TS-map pipelines, enabling quicker, more precise follow-up by optical and infrared surveys.


# Results with Toy Data

Resolution = 5 deg
No of Compton Events | No of Background Events | Mean Angular Deviation | Median Angular Deviation | RMS Angular Deviation 
--- | --- | --- | --- | --- 
2000 | 0 | 93.91 | | 
1333 | 667 | 93.91 | | 
1000 | 1000 | 93.91 | | 


Resolution = 3 deg
No of Compton Events | No of Background Events | Mean Angular Deviation | Median Angular Deviation | RMS Angular Deviation 
--- | --- | --- | --- | --- 
2000 | 0 | 93.91 | | 
1333 | 667 | 93.91 | | 
1000 | 1000 | 93.91 | | 

# Results with MEGAlib Simulated Data

Resolution = 5 deg
No of Compton Events | No of Background Events | Mean Angular Deviation | Median Angular Deviation | RMS Angular Deviation 
--- | --- | --- | --- | --- 
2000 | 0 | 93.91 | | 
1333 | 667 | 93.91 | | 
1000 | 1000 | 93.91 | | 


Resolution = 3 deg
No of Compton Events | No of Background Events | Mean Angular Deviation | Median Angular Deviation | RMS Angular Deviation 
--- | --- | --- | --- | --- 
2000 | 0 | 93.91 | | 
1333 | 667 | 93.91 | | 
1000 | 1000 | 93.91 | | 




## Results

> [!NOTE]
> **Binned vs. Unbinned.**  
> *Binned* = histograms constructed from events; *Unbinned* = per-event likelihood (no histograms).  
> All angles are in **degrees (°)**.

### Toy Data

<details>
  <summary><b>▶ Binned results (shown below)</b></summary>

**Resolution = 5°**

No. Compton | No. Background | Mean Angular Dev. | Median Angular Dev. | RMS Angular Dev.
:--:|:--:|:--:|:--:|:--:
2000 | 0    | 93.91 |  | 
1333 | 667  | 93.91 |  | 
1000 | 1000 | 93.91 |  | 

**Resolution = 3°**

No. Compton | No. Background | Mean Angular Dev. | Median Angular Dev. | RMS Angular Dev.
:--:|:--:|:--:|:--:|:--:
2000 | 0    | 93.91 |  | 
1333 | 667  | 93.91 |  | 
1000 | 1000 | 93.91 |  | 

</details>

<details>
  <summary><b>▶ Unbinned results</b></summary>

> [!TIP]
> Replace the placeholders below with your unbinned metrics once computed.

**Resolution = 5°**

No. Compton | No. Background | Mean Angular Dev. | Median Angular Dev. | RMS Angular Dev.
:--:|:--:|:--:|:--:|:--:
2000 | 0    | — | — | — 
1333 | 667  | — | — | — 
1000 | 1000 | — | — | — 

**Resolution = 3°**

No. Compton | No. Background | Mean Angular Dev. | Median Angular Dev. | RMS Angular Dev.
:--:|:--:|:--:|:--:|:--:
2000 | 0    | — | — | — 
1333 | 667  | — | — | — 
1000 | 1000 | — | — | — 

</details>

---

### MEGAlib Simulated Data

<details>
  <summary><b>▶ Binned results (shown below)</b></summary>

**Resolution = 5°**

No. Compton | No. Background | Mean Angular Dev. | Median Angular Dev. | RMS Angular Dev.
:--:|:--:|:--:|:--:|:--:
2000 | 0    | 93.91 |  | 
1333 | 667  | 93.91 |  | 
1000 | 1000 | 93.91 |  | 

**Resolution = 3°**

No. Compton | No. Background | Mean Angular Dev. | Median Angular Dev. | RMS Angular Dev.
:--:|:--:|:--:|:--:|:--:
2000 | 0    | 93.91 |  | 
1333 | 667  | 93.91 |  | 
1000 | 1000 | 93.91 |  | 

</details>

<details>
  <summary><b>▶ Unbinned results</b></summary>

**Resolution = 5°**

No. Compton | No. Background | Mean Angular Dev. | Median Angular Dev. | RMS Angular Dev.
:--:|:--:|:--:|:--:|:--:
2000 | 0    | — | — | — 
1333 | 667  | — | — | — 
1000 | 1000 | — | — | — 

**Resolution = 3°**

No. Compton | No. Background | Mean Angular Dev. | Median Angular Dev. | RMS Angular Dev.
:--:|:--:|:--:|:--:|:--:
2000 | 0    | — | — | — 
1333 | 667  | — | — | — 
1000 | 1000 | — | — | — 

</details>

---

#### Metric definitions

- **Mean Angular Deviation**: \(\frac{1}{N}\sum_i \Delta\theta_i\)  
- **Median Angular Deviation**: 50th percentile of \(\Delta\theta\)  
- **RMS Angular Deviation**: \(\sqrt{\frac{1}{N}\sum_i \Delta\theta_i^2}\)

> [!IMPORTANT]
> Reported numbers above in **Binned** sections are placeholders you supplied; fill in missing **Median**/**RMS** cells as you compute them.  
> Keep resolutions grouped to make comparisons easy across background levels.
