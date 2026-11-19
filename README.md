# Hydroxyl (OH) maser W3(OH) from public Allen Telescope Array IQ data

This project processes a public raw-IQ recording of the 1665 MHz hydroxyl (OH) line toward the star-forming region W3(OH). One script, `oh_spectrum.py`, takes the raw samples all the way to physical results:

- an averaged, baseline-corrected spectrum with an LSR velocity axis, and a dynamic spectrum (waterfall) that removes instrumental gain jumps (`spectrum`);
- full polarization (Stokes I, Q, U, V) from cross-correlating the two feeds (`stokes`);
- a Gaussian decomposition of the line profile, a noise check against the radiometer equation, and a kinematic distance (`fit`).

A second script, `make_synthetic_test.py`, generates a synthetic recording with known properties to validate the pipeline.

**Dataset:** Daniel Estévez, *IQ recording of W3OH hydroxyl spectral line at 1665.4 MHz*, Zenodo record 6711078, published 25 June 2022 (v1, open):
<https://zenodo.org/records/6711078>

---

## Contents

1. [The dataset](#1-the-dataset)
2. [Project layout](#2-project-layout)
3. [Setup](#3-setup)
4. [Commands at a glance](#4-commands-at-a-glance)
5. [Running the analysis](#5-running-the-analysis)
6. [What the outputs show](#6-what-the-outputs-show)
7. [Validation on synthetic data](#7-validation-on-synthetic-data)
8. [Results](#8-results)
9. [Caveats and limitations](#9-caveats-and-limitations)
10. [Possible next steps](#10-possible-next-steps)
11. [Change log](#11-change-log)

---

## 1. The dataset

- **Target.** The OH radical spectral line (rest frequency **1665.4018 MHz**) toward the W3(OH) complex, recorded with one of the **6.2 m antennas of the Allen Telescope Array (ATA)** at Hat Creek, California. The recording lasts about 500 seconds and is stored as **raw IQ voltage samples**, not spectra.
  - W3(OH) is a compact HII region in the W3 complex, in the Perseus Arm of the Milky Way. Its OH emission comes mostly from **masers**, which appear as very narrow, bright spectral spikes rather than the broad hump of the hydrogen (HI) line.
- **Pointing (J2000).** RA 02h 27m 04.1s, Dec +61° 52′ 22″.
- **Signal chain.**
  - Antenna 1a feeds a downconverter tuned to 1665.4 MHz, with a 512 MHz intermediate-frequency (IF) output.
  - A **USRP N321** digitizes both **linear polarizations (X and Y)**, using a shared local oscillator (LO). Because the two channels share the LO, they can be cross-correlated, which is what the `stokes` command does.
  - The USRP is locked to the observatory's external 10 MHz reference and pulse-per-second (PPS) signal, so the frequency axis is accurate. That matters when measuring Doppler shifts.
  - A GNU Radio flowgraph wrote the samples to disk.
- **Sample rate.** 1.92 Msps (complex), so the recording covers 1.92 MHz of bandwidth.
- **Format.** 16-bit integer IQ (SigMF datatype `ci16_le`), originally in GNU Radio's metadata format with detached headers, then converted to **SigMF**. (The Zenodo page says "16-bin", which is almost certainly a typo for 16-bit.)
- **Files.** 7.8 GB total: one 3.9 GB `.sigmf-data` file and one 920-byte `.sigmf-meta` file per polarization:
  ```
  W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-data   (3.9 GB)
  W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta   (920 B)
  W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-data   (3.9 GB)
  W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-meta   (920 B)
  ```
  - The sizes are consistent: 1.92 Msps × 4 bytes per complex sample × ~500 s ≈ 3.84 GB.
  - The script reads **971,942,220 samples = 506.2 s** per polarization.
  - The metadata gives the start as **2022-06-21 19:36:55.16**, which the script treats as UTC.
- **Bandwidth caveat.** 1.92 MHz centered on 1665.4 MHz covers about **1664.44–1666.36 MHz**. W3(OH) also has a strong OH line at 1667.359 MHz, but it falls **outside** this recording. Only the 1665 MHz line is present.

---

## 2. Project layout

```
OH Spectral Line Analysis/
├── iq_data/
│   ├── W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-data
│   ├── W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta
│   ├── W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-data
│   └── W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-meta
├── validation/
│   ├── make_synthetic_test.py
│   └── synth/                  (created by make_synthetic_test.py: PX/PY SigMF pairs, ~230 MB each)
├── oh_spectrum.py
├── README.md
└── (outputs: *.png, *.csv)
```

- Each `.sigmf-data` file must sit next to its `.sigmf-meta` file with the same base name. You can pass either file to the script and it finds the other.
- Outputs are saved in the folder you run the command from, not next to the data.
- When running from `validation/`, the script is one folder up: `python ..\oh_spectrum.py ...`.

---

## 3. Setup

### Libraries

| Library | Used for | Needed by |
|---|---|---|
| **Python** 3.8+ | | everything |
| **numpy** | array math, FFTs | everything |
| **scipy** | running-median baseline (`ndimage`), Gaussian fitting (`optimize`), synthetic signal filters (`signal`) | everything; `scipy.signal` only by `make_synthetic_test.py` |
| **matplotlib** | plots | `spectrum`, `stokes`, `fit` |
| **astropy** | LSR velocity correction, Galactic coordinates | optional: without it velocities stay topocentric and `fit` skips the kinematic plot |

### Recommended: a separate conda environment

A dedicated environment avoids version conflicts with other packages:

```
conda create -n radio -c conda-forge python=3.12 numpy scipy matplotlib astropy inspectrum
conda activate radio
```

In VS Code, press Ctrl+Shift+P, choose **Python: Select Interpreter**, pick `radio`, then open a new terminal. Run `conda activate radio` before each session.

### Troubleshooting

- **`ModuleNotFoundError: No module named 'numpy'`** (or another library): the terminal is using a different Python from the one you installed into. Select the `radio` interpreter as above.
- **`ImportError: cannot import name '_promote' from 'scipy.spatial.transform._rotation'`**: the SciPy installation is broken, usually by a partial upgrade while another Python process held its files open. Close VS Code, then reinstall SciPy (`conda install -c conda-forge --force-reinstall scipy`, or `pip install --force-reinstall --no-cache-dir scipy` if `conda list scipy` shows `pypi`), or create the `radio` environment above.
- **`can't open file ... oh_spectrum.py`**: you are in a different folder from the script. Use `..\oh_spectrum.py` from a subfolder, or `cd` to the script's folder.

### Optional viewers

- **inspectrum** opens `.sigmf-data` directly as a spectrogram. Each column is a single FFT with no averaging, so the weak maser is hard or impossible to see there.
- **SDR#** can play the data after conversion to WAV (`wav` command below).

---

## 4. Commands at a glance

| Command | Input | What it does | Outputs |
|---|---|---|---|
| `info` | one `.sigmf-meta` | prints sample rate, center frequency, start time, duration and the full metadata | console only |
| `spectrum` | one or more `.sigmf-meta` (one per polarization) | averaged spectrum, baseline removal, LSR velocity, per-row normalized waterfall, row-gain plot | `<prefix>.png`, `<prefix>.csv` |
| `stokes` | X then Y `.sigmf-meta` | cross-correlates X and Y; Stokes I, Q, U, V; per-feature polarization fractions and X–Y phase | `<prefix>.png`, `<prefix>.csv` |
| `fit` | a CSV written by `spectrum` | Gaussian components (number chosen by BIC), noise vs radiometer equation, thermal widths, kinematic distance | `<prefix>.png`, `<prefix>_kinematic.png`, `<prefix>.csv` |
| `wav` | one `.sigmf-meta` | converts to a WAV file SDR# can play | `SDRSharp_..._IQ.wav` |

Run any command with `-h` for every option, for example `python oh_spectrum.py fit -h`.

---

## 5. Running the analysis

Run these from the `OH Spectral Line Analysis` folder. Each command is a single line.

### Step 1: check the metadata

```
python oh_spectrum.py info iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta
```

### Step 2: averaged spectrum and waterfall, both polarizations (`xy_rows`)

```
python oh_spectrum.py spectrum iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-meta --vmin -50 --vmax -40 --out xy_rows
```

About 4 minutes for both files (each file is read separately). Add `--seconds 60` for a quick test on the first minute.

### Step 3: polarization (`w3oh_stokes`)

```
python oh_spectrum.py stokes iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-meta --out w3oh_stokes
```

X must come first, then Y. About 4–5 minutes, because both files are read in lockstep.

### Step 4: components, noise check and kinematic distance (`xy_rows_fit`)

```
python oh_spectrum.py fit xy_rows.csv --integration 506.2
```

Needs the CSV from step 2: a two-file `spectrum` run with the LSR correction applied. Takes seconds.

### Step 5 (optional): WAV for SDR#

```
python oh_spectrum.py wav iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta
```

In SDR#, choose **Source → IQ File (\*.wav)**. If the spectrum looks mirrored, add `--swap-iq`.

### Main options

**Shared by `spectrum` and `stokes`**

| Option | Default | Meaning |
|---|---|---|
| `--nfft` | 16384 | FFT size. Resolution = 1.92 MHz / nfft (16384 → 117 Hz, 4096 → 469 Hz) |
| `--seconds`, `--skip` | all, 0 | process only part of the recording |
| `--vmin`, `--vmax` | −100, +10 (`spectrum`); −50, −40 (`stokes`) | velocity window in km/s. **Must match the velocity frame**: LSR if astropy is installed, topocentric if not |
| `--baseline-khz` | 300 | running-median width for the bandpass. Must be much wider than the line |
| `--edge`, `--dc-mask` | 0.06, 1500 Hz | band-edge fraction and DC-spike half-width to exclude |
| `--rest` | 1665.4018e6 | rest frequency (Hz). Use 1420.405751768e6 for hydrogen |
| `--ra`, `--dec`, `--site` | W3(OH), ATA | target and observatory |
| `--start`, `--center`, `--rate` | from metadata | overrides |
| `--no-lsr` | off | keep velocities topocentric |
| `--out`, `--no-show` | | output prefix; save without opening a window |

**`spectrum` only**

| Option | Default | Meaning |
|---|---|---|
| `--dump` | 10 | seconds per waterfall row |
| `--row-norm` | `scale` | `scale`: each row rescaled to its own gain (removes gain jumps). `median`: each row divided by its own running median (also follows bandpass-shape changes; slower). `none`: single time-averaged baseline (original behaviour) |

**`stokes` only**

| Option | Default | Meaning |
|---|---|---|
| `--features` | automatic | feature velocities to measure; otherwise peaks above `--min-snr` |
| `--min-snr` | 6 | automatic peak threshold (σ, after 5-channel smoothing) |
| `--halfwidth` | 0.15 | half-width of each feature's window (km/s) |
| `--xy-phase` | none | instrumental phase the receiver adds to XY\*, in degrees; it is subtracted so U and V can be separated. Unknown for this dataset |

**`fit`**

| Option | Default | Meaning |
|---|---|---|
| `--column` | `mean_line` | profile to fit; e.g. a single polarization's `..._line` column |
| `--vmin`, `--vmax` | −48, −42.5 | fit window (km/s) |
| `--guess` | −46.2 −45.2 −44.9 −44.1 | starting component velocities |
| `--max-extra` | 1 | extra components tried at the largest residual, kept only if the BIC improves |
| `--center-range` | 0.4 | how far a component may move from its guess (km/s) |
| `--min-fwhm`, `--max-fwhm` | 0.07, 2.5 | allowed widths (km/s); components at a limit are flagged |
| `--noise` | −60 −50 −40 −30 | line-free velocity ranges, as pairs |
| `--integration` | none | integration time (s), for the radiometer comparison |
| `--mass-amu`, `--tkin` | 17; 50 100 150 | molecule mass and temperatures for the thermal-width comparison |
| `--r0`, `--theta0` | 8.15, 236 | rotation curve (Reid et al. 2019) |
| `--parallax` | 1.95 | distance to compare (Xu et al. 2006); 0 to omit |
| `--kin-velocity` | strongest component | velocity used for the kinematic distance |
| `--no-kinematic` | off | skip the kinematic plot |

---

## 6. What the outputs show

### `spectrum` (e.g. `xy_rows.png`)

1. **Raw averaged spectrum** (dB vs frequency offset): the receiver bandpass, the DC spike at the center and the maser bump at about +335 kHz. Dashed lines are the estimated baselines.
2. **Baseline-removed line profile** vs LSR velocity, for each polarization and their mean, as `(S − baseline) / baseline`, the line power as a fraction of the system noise.
3. **Dynamic spectrum**: mean of the polarizations, with each row normalized by its own gain.
4. **Row gain vs time** for each polarization: the broadband gain changes that step 3 removes.

The CSV holds frequency, velocity, raw power and line profile per file, plus the mean. The console reports the LSR correction, the strongest feature and the range of row gains.

### Why panel 1 is transformed into panel 2

The power measured in each frequency bin is approximately:

```
measured(f) = gain(f) × [system noise + line(f)]
```

The receiver gain varies across the band and is not known, and the line is a few percent of the system noise. A running median (300 kHz wide) ignores narrow features and estimates `baseline(f) ≈ gain(f) × system noise`. Dividing out the baseline cancels the gain:

```
measured / baseline − 1 = line(f) / system noise
```

The frequency axis is then converted to velocity using the radio Doppler convention, `v = c (f₀ − f) / f₀`, and corrected to the local standard of rest (LSRK). The correction adds Earth's rotation and orbit (the barycentric correction, from astropy) and the Sun's standard 20 km/s motion. It was cross-checked against astropy's independent `SpectralCoord` method and agreed to within 0.003 km/s.

### Per-row normalization (panels 3 and 4)

Each 10 s row is divided by the time-averaged bandpass, then rescaled by that row's own gain: the median ratio of row to bandpass over all usable channels. The median ignores the few channels with maser emission, so a broadband gain jump is removed while the line is kept. A short final row (fewer than half the usual FFTs) is dropped, because the median of only a few averaged spectra is biased low.

### `stokes` (e.g. `w3oh_stokes.png`)

For linear feeds X and Y:

```
I = XX + YY        Q = XX − YY        U = 2 Re(XY*)        V = 2 Im(XY*)
```

Each auto-power is divided by its own bandpass, and the cross-power has its own baseline (correlated receiver noise) subtracted and is divided by √(Bx·By). Every Stokes parameter is therefore a fraction of one channel's system noise.

1. **I and Q.** Q shows linear polarization along X (+) or Y (−).
2. **The cross-correlation.** `U'` and `V'` are its real and imaginary parts. Without an X–Y phase calibration they are mixtures of U and V. Their magnitude, `√(U² + V²) = 2|XY*|`, does not depend on the phase.
3. **The X–Y phase** of each channel with significant cross-power, plus each feature's summed value (black).

The console table gives, per feature:

| Column | Meaning |
|---|---|
| `q = Q/I` | X vs Y difference. Non-zero at > 3σ means linear polarization along a feed (assumes equal system temperatures in X and Y) |
| `sqrt(U2+V2)/I` | fraction of power correlated between the feeds: linear at ±45° and/or circular. Independent of the unknown X–Y phase. Noise-bias corrected |
| `X-Y phase` | includes an unknown instrumental offset, so only **differences** between features mean anything. Features about 180° apart have opposite-sign polarization; for purely circular lines that means opposite hands, the signature of a Zeeman pair |

Uncertainties are measured empirically from line-free channels, so they include the correlation between neighbouring channels, and they include the noise in I.

### `fit` (e.g. `xy_rows_fit.png`, `xy_rows_fit_kinematic.png`)

- **Components plot:** the measured profile, each Gaussian component, their sum, and the residuals in units of σ.
- **Kinematic plot:** the LSR velocity expected from circular Galactic rotation along the line of sight, the measured velocity, the resulting kinematic distance, and the parallax distance for comparison.
- **Console and CSV:** each component's velocity, peak, FWHM and area. Uncertainties come from the fit covariance and are lower limits, because neighbouring channels are correlated.

---

## 7. Validation on synthetic data

`make_synthetic_test.py` builds a 30 s two-polarization SigMF recording in the same format as the ATA data, with known properties. The random seed is fixed, so every run produces byte-identical files (about 230 MB each). The properties are listed at the top of the script and in the `.sigmf-meta` annotations, which inspectrum displays.

| Property | Injected | Recovered |
|---|---|---|
| Y/X power ratio | 2.92 dB | removed by normalization |
| X gain jump (10–15 s) | +3% amplitude (+6.09% power) | +6.06% |
| Y gain variation | none | −0.10% to +0.14% |
| Instrumental X–Y phase | −40° on XY\* | phases offset by −36° to −42° |
| Line A (−42.88 km/s) | linear along X | q = +0.972 ± 0.050 |
| Line E (−43.78 km/s) | linear at 45° | √(U²+V²)/I = 0.974 ± 0.053, phase −36° (expected −40°) |
| Line B (−44.68 km/s) | circular | √(U²+V²)/I = 1.129 ± 0.062, phase −132° (expected −130°) |
| Line C (−45.58 km/s) | circular, opposite hand | √(U²+V²)/I = 0.976 ± 0.051, phase +49° (expected +50°); 181° from B |
| Line D (−46.48 km/s) | unpolarized | q = +0.004 ± 0.016, √(U²+V²)/I = 0.021 ± 0.020 |
| Correlated X–Y noise | none | 0.0001 of system power |

Every measured value agrees with its injected value to within about 2σ.

To reproduce, from the `validation` folder:

```
python make_synthetic_test.py --out-dir synth
python ..\oh_spectrum.py spectrum synth/PX.sigmf-meta synth/PY.sigmf-meta --no-lsr --vmin -48 --vmax -41 --dump 2
python ..\oh_spectrum.py stokes synth/PX.sigmf-meta synth/PY.sigmf-meta --no-lsr --vmin -48 --vmax -41
```

Adding `--xy-phase -40` to the `stokes` command removes the injected phase offset. U and V are then separated: line E appears in U, and lines B and C appear in V with opposite signs.

---

## 8. Results

LSR correction at mid-recording (2022-06-21 19:41 UTC): **+14.94 km/s**. Topocentric velocity + 14.94 = LSR velocity.

### Detection and noise

- OH maser emission between about **−46.5 and −44 km/s (LSRK)**.
- Strongest channel: **−45.17 km/s** (1665.7357 MHz), 0.049 of the system noise, about **16σ**.
- Line-free noise in the X+Y average: **0.00270**, against **0.00290** from the radiometer equation (117.2 Hz × 506.2 s × 2 polarizations), a ratio of **0.93**. The spectrum is limited by thermal noise.

### Instrumental gain variations

- **X:** −2.5% to +6.1%, with abrupt steps at about 200 s and 370 s and a plateau until about 460 s.
- **Y:** −0.6% to +0.9%, a slow, smooth drift.
- Abrupt steps in one channel only point to the X-channel electronics rather than the sky or the pointing, although polarized interference cannot be fully ruled out.
- After per-row normalization the maser is a steady vertical stripe for the full 506 s, with no variability detectable in 10 s rows.

### Line components (`fit`, X+Y average)

Five components fit to within the noise: reduced χ² **0.98**. The BIC prefers five components over four (325.3 against 331.2).

| v (km/s) | Peak | FWHM (km/s) | Area (km/s) | Note |
|---|---|---|---|---|
| −46.220 ± 0.008 | 0.0209 ± 0.0011 | 0.313 ± 0.022 | 0.00696 | narrow |
| −45.300 ± 0.082 | 0.0080 ± 0.0013 | 1.672 ± 0.220 | 0.01428 | broad wing, 32% of the total area |
| −45.197 ± 0.006 | 0.0376 ± 0.0014 | 0.481 ± 0.020 | 0.01925 | main component |
| −44.130 ± 0.021 | 0.0079 ± 0.0011 | 0.345 ± 0.060 | 0.00289 | weak |
| −42.681 ± 0.018 | 0.0056 ± 0.0015 | 0.145 ± 0.043 | 0.00087 | tentative (~3.7σ) |

- **The "shoulder" near −44.9 km/s** from the earlier analysis is better described by the broad component. That component is plausibly many weak maser spots blended in the beam, but the data cannot confirm this.
- **The −42.68 km/s component is tentative.** Fitted per polarization, it appears in Y (~3.9σ) but not in X, and it is below the 6σ threshold of the polarization analysis.
- **Widths compared with thermal broadening.** The thermal FWHM of OH is 0.37, 0.52 and 0.64 km/s at 50, 100 and 150 K. If purely thermal, the narrow components would need gas colder than about 36 K (−46.22), 44 K (−44.13) and 8 K (−42.68), which supports maser amplification.
- **The main component is probably a blend.** At 0.48 km/s it is wider than the 0.15–0.38 km/s of individual W3(OH) maser spots measured with the VLBA (Fish et al. 2006).

### Polarization (`stokes`)

Two features detected above 6σ:

| v (km/s) | I S/N | q = Q/I | √(U²+V²)/I | X–Y phase |
|---|---|---|---|---|
| −46.18 | 19 | +0.062 ± 0.049 | 1.150 ± 0.082 | −172° ± 3° |
| −45.19 | 39 | −0.184 ± 0.024 | 0.196 ± 0.027 | −75° ± 8° |

- **−46.18 km/s: essentially fully polarized,** but equally strong in X and Y (q ≈ 0). A strongly circularly polarized maser would look like this. So would linear polarization at 45° to the feeds, which cannot be distinguished without a phase calibration.
- **−45.19 km/s (main): significantly stronger in Y** (q = −0.18, 7.7σ). The −46.18 km/s feature, with q ≈ 0, acts as a control: a simple system-temperature difference would give every feature the same q. The total polarized fraction is at least √(0.18² + 0.20²) ≈ 0.27.
- **The phase sweeps across the main feature,** from about 180° on its blue side (−45.5 km/s) through −90° to about 0° on its red side (−45.0 km/s). The main feature is a blend of components with different polarization, which partly cancel when summed. Its 0.20 correlated fraction is therefore a lower limit for the individual components.
- **Possible Zeeman pair (hypothesis only).** The red side of the main feature is about 180° from the −46.2 km/s feature. If both are mainly circular, they could be a Zeeman pair about 1.2 km/s apart, roughly 2 mG at about 0.59 km/s per mG. A single dish cannot show they come from the same maser spot, and linear polarization could produce the same phase difference.
- **Correlated noise between the feeds:** 0.0525 of the system power, removed as a baseline. The synthetic test showed 0.0001, so this correlation is instrumental.

### Kinematic distance (`fit`)

- Position: (l, b) = (133.95°, 1.06°). Flat rotation curve with R₀ = 8.15 kpc and Θ₀ = 236 km/s (Reid et al. 2019).
- For the main component, −45.20 km/s: **kinematic distance 3.77 kpc**. Using any other narrow component gives 3.67–3.87 kpc.
- At the **parallax distance of 1.95 kpc** (Xu et al. 2006), circular rotation predicts **−25.8 km/s**.
- The ~**19 km/s** difference is not a measurement error. It reproduces the anomalous (non-circular) motion of this part of the Perseus Arm that the parallax measurement revealed.

---

## 9. Caveats and limitations

- **No flux calibration.** All intensities are relative to the system noise, not kelvins or janskys, because the system temperature of this antenna is not known here. Professional observatories calibrate against known hot and cold sources.
- **Single dish.** The ~2° beam blends emission from the whole region. Features cannot be assigned to individual maser spots, and some fitted components are probably blends.
- **Uncalibrated polarization.**
  - U and V cannot be separated, so circular and 45°-linear polarization look the same.
  - q assumes equal system temperatures in X and Y. The −46.2 km/s feature, with q ≈ 0, partly guards against this.
  - If the 5% correlated noise comes from coupling between the feeds, it could add spurious polarization of order 0.05 of I. That is small compared with the −46.2 km/s feature's 1.15, but about a quarter of the main feature's 0.20.
- **Fit uncertainties.** The Gaussian decomposition is not unique, its uncertainties are lower limits, and single-polarization fits are less stable than fits to the X+Y average.
- **Bandwidth.** The 1667 MHz line is outside the recording, so the 1665/1667 ratio cannot be measured.
- **Time reference.** The start time is taken as UTC. A clock error of a few hours would change the LSR correction by at most about 0.2 km/s (0.4 km/s for 12 hours), well below the line widths.
- **Single short epoch.** 8.4 minutes on one date cannot test for the month-to-year variability typical of masers.
- **Simple Galactic model.** A flat rotation curve and the standard solar motion. Other models would shift the 3.77 kpc result somewhat, but not enough to remove the ~19 km/s discrepancy.
- **Velocity window must match the frame.** If astropy is missing, velocities are topocentric (about 15 km/s more negative here), and LSR-based `--vmin`/`--vmax` values will miss the line. The `fit` command also expects an LSR CSV.

---

## 10. Possible next steps

- **Split-half test:** process 0–250 s and 250–506 s separately. Real features should match; instrumental ones won't.
- **Per-feature light curves** from the waterfall rows, for X and Y separately, to check stability and whether the X gain jumps leak into the line.
- **Noise vs integration time** on a log-log plot. It should fall as 1/√t if the data stays noise-limited.
- **Compare with a published W3(OH) 1665 MHz spectrum** and other epochs (for example the Stockert observations discussed on the SARA mailing list) to look for variability.
- **Higher resolution** (`--nfft 65536`, about 29 Hz) to see whether the main component splits into narrower parts.
- **Apply the pipeline to your own recordings**, for example RTL-SDR hydrogen-line data: `--rest 1420.405751768e6` plus your own `--ra`, `--dec` and `--site`. For single-polarization data, use `spectrum` and `fit` (with `--column`, `--guess` and `--vmin/--vmax` set for that line).

---

## 11. Change log

**November 16–17, 2026**
- `spectrum`: per-row normalization of the waterfall (`--row-norm`), X+Y averaged waterfall, new row-gain panel, short final row dropped.
- New `stokes` command: X–Y cross-correlation, Stokes parameters, per-feature polarization fractions with noise-bias correction and empirical uncertainties.
- New `fit` command (replaces the separate `fit_components.py` and `plots.py`): Gaussian decomposition with BIC selection, radiometer noise check, thermal widths, kinematic-distance plot.
- New `make_synthetic_test.py` for validation.
- `stokes` no longer needs `scipy.signal`.

**November 13, 2026**
- First version: averaged spectrum, baseline removal, LSR correction, dynamic spectrum, WAV export.
- Earlier runs, kept for reference:
  - **60 s quick test (X only):** first detection at about −60 km/s topocentric, ~4σ.
  - **`x_full` (X, 469 Hz bins, topocentric):** reported −59.92 km/s at 0.023, ~12σ. The main peak (−60.1 km/s topocentric) was just outside the `--vmin -60` window, so this was the line's edge, not its peak. Lesson: the velocity window must match the velocity frame.
  - **`x_fine` (X, 117 Hz bins, LSRK):** first resolved structure: −45.2 km/s (0.043) and −46.2 km/s (0.027), plus the bright horizontal bands from the X-channel gain jumps, later removed by per-row normalization.

---

## References

- D. Estévez, *IQ recording of W3OH hydroxyl spectral line at 1665.4 MHz*, Zenodo record 6711078 (2022): <https://zenodo.org/records/6711078>
- Y. Xu, M. J. Reid, X. W. Zheng, K. M. Menten, *The Distance to the Perseus Spiral Arm in the Milky Way*, Science 311, 54 (2006): <https://doi.org/10.1126/science.1120914>
- M. J. Reid et al., *Trigonometric Parallaxes of High-mass Star-forming Regions: Our View of the Milky Way*, ApJ 885, 131 (2019): <https://doi.org/10.3847/1538-4357/ab4a11>
- V. L. Fish, W. F. Brisken, L. O. Sjouwerman, *A Very High Spectral Resolution Study of Ground-State OH Masers in W3(OH)*, ApJ 647, 418 (2006): <https://doi.org/10.1086/505420>
- E. White, S. Croft, A. W. Pollak, D. Kozel, *GNU Radio at the Allen Telescope Array*, Proceedings of the GNU Radio Conference 5(1) (2020): <https://pubs.gnuradio.org/index.php/grcon/article/view/70>
- SigMF, the Signal Metadata Format: <https://github.com/sigmf/SigMF>
