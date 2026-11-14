# Hydroxyl (OH) maser W3(OH) from public Allen Telescope Array IQ data

This project processes a public raw-IQ recording of the 1665 MHz hydroxyl (OH) line toward the star-forming region W3(OH). It turns the raw samples into an averaged, baseline-corrected spectrum with a Doppler velocity axis, using `oh_spectrum.py`.

**Dataset:** *IQ recording of W3OH hydroxyl spectral line at 1665.4 MHz*, Zenodo record 6711078, published 25 June 2022 (v1, open):
<https://zenodo.org/records/6711078>

---

## 1. The dataset

### What's in it

- **Target.** The OH radical spectral line (rest frequency **1665.4018 MHz**) toward the W3(OH) complex, recorded with one of the **6.2 m antennas of the Allen Telescope Array (ATA)** at Hat Creek, California. The recording lasts about 500 seconds and is stored as **raw IQ voltage samples**, not spectra.
  - W3(OH) is a star-forming region in the Perseus Arm of the Milky Way. Its OH emission comes mostly from **masers**, which appear as very narrow, bright spectral spikes rather than the broad hump of the hydrogen (HI) line.
- **Pointing (J2000).** RA 02h 27m 04.1s, Dec +61° 52′ 22″.
- **Signal chain.**
  - Antenna 1a feeds a downconverter tuned to 1665.4 MHz, with a 512 MHz intermediate-frequency (IF) output.
  - A **USRP N321** digitizes both **linear polarizations (X and Y)**, using a shared local oscillator (LO).
  - The USRP is locked to the observatory's external 10 MHz reference and pulse-per-second (PPS) signal, so the frequency axis is accurate. That matters when measuring Doppler shifts.
  - A GNU Radio flowgraph wrote the samples to disk.
- **Sample rate.** 1.92 Msps (complex), so the recording covers 1.92 MHz of bandwidth.
- **Format.** 16-bit integer IQ, originally in GNU Radio's metadata format with detached headers, then converted to **SigMF**. (The Zenodo page says "16-bin", which is almost certainly a typo for 16-bit.)
- **Files.** 7.8 GB total: one 3.9 GB `.sigmf-data` file and one 920-byte `.sigmf-meta` file per polarization:
  ```
  W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-data   (3.9 GB)
  W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta   (920 B)
  W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-data   (3.9 GB)
  W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-meta   (920 B)
  ```
  - The sizes are consistent: 1.92 Msps × 4 bytes per complex sample × ~500 s ≈ 3.84 GB.
  - The script reads **971,942,220 samples = 506.2 s** per polarization.
  - The file names give the start as **2022-06-21 19:36:55**. UTC is assumed.
- **Bandwidth caveat.** 1.92 MHz centered on 1665.4 MHz covers about **1664.44–1666.36 MHz**. W3(OH) also has a strong OH line at 1667.359 MHz, but it falls **outside** this recording. Only the 1665 MHz line is present.

---

## 2. Getting the IQ data

**The IQ recordings are not included in this repository.** At 7.8 GB they are far too large for GitHub, so you need to download them yourself:

1. Go to the Zenodo record: <https://zenodo.org/records/6711078>
2. Download all four files (both `.sigmf-data` and both `.sigmf-meta`).
3. In the project folder, create a folder named `iq_data` and move the four files into it.

## 3. Project layout

This repository contains only the script and this README:

```
Hydroxyl-public-data/
├── oh_spectrum.py
└── README.md
```

After you download the data and run the script, the folder should look like this:

```
Hydroxyl-public-data/
├── iq_data/                 ← you create this and add the downloaded files
│   ├── W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-data
│   ├── W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta
│   ├── W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-data
│   └── W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-meta
├── oh_spectrum.py
├── README.md
└── (outputs: *.png, *.csv)
```

Each `.sigmf-data` file must sit next to its `.sigmf-meta` file with the same base name. You can pass either file to the script and it finds the other.

---

## 4. Requirements

- **Python** 3.8 or newer
- **numpy**: array math and FFTs
- **scipy**: running-median baseline (`scipy.ndimage.median_filter`)
- **matplotlib**: plots
- **astropy**: correction to the local standard of rest (LSR). This is optional: without it the script still runs, but velocities are **topocentric** (as seen from Earth, uncorrected).

Install everything with conda:

```
conda install -c conda-forge numpy scipy matplotlib astropy
```

or with pip:

```
pip install numpy scipy matplotlib astropy
```

> **VS Code note:** if the terminal reports `ModuleNotFoundError`, it is using a different Python from the one you installed into. Press Ctrl+Shift+P, choose **Python: Select Interpreter**, pick the `miniconda3` environment, then open a new terminal.


---

## 5. Running the script

Run all commands from the project folder. They assume you've downloaded the IQ files from [Zenodo](https://zenodo.org/records/6711078) into `iq_data\` as described in section 2. If you saved them somewhere else, replace `iq_data\` in each command with the path to your copy.

The paths use Windows backslashes (`\`). On macOS or Linux, use forward slashes (`iq_data/...`).

### Inspect the metadata

```
python oh_spectrum.py info iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta
```

Prints sample rate, center frequency, start time, datatype and duration, plus the full JSON.

### Quick test (first 60 s, X only)

```
python oh_spectrum.py spectrum iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta --seconds 60
```

### Full run, X only, coarse bins (`x_full`)

```
python oh_spectrum.py spectrum iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta --vmin -60 --vmax -30 --nfft 4096 --out x_full
```

### Full run, X only, fine bins (`x_fine`)

```
python oh_spectrum.py spectrum iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta --vmin -50 --vmax -40 --out x_fine
```

### Both polarizations, fine bins (`xy_fine`)

```
python oh_spectrum.py spectrum iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_y.sigmf-meta --vmin -50 --vmax -40 --out xy_fine
```

### Convert to WAV for SDR#

```
python oh_spectrum.py wav iq_data\W3OH_1665MHz_2022-06-21T19_36_55_pol_x.sigmf-meta
```

This writes `SDRSharp_<date>_<time>Z_1665400000Hz_IQ.wav`. In SDR#, choose **Source → IQ File (\*.wav)**. If the spectrum looks mirrored, add `--swap-iq`.

### Main `spectrum` options

| Option | Default | Meaning |
|---|---|---|
| `--nfft` | 16384 | FFT size. Resolution = 1.92 MHz / nfft (16384 → 117 Hz/bin, 4096 → 469 Hz/bin) |
| `--seconds` | all | Process only this many seconds |
| `--skip` | 0 | Skip this many seconds at the start |
| `--vmin`, `--vmax` | −100, +10 | Velocity window for panels 2 and 3 (km/s). **Must match the velocity frame**: LSR if astropy is installed, topocentric if not |
| `--baseline-khz` | 300 | Running-median width for the bandpass estimate. Must be much wider than the line |
| `--dump` | 10 | Seconds per row of the dynamic spectrum |
| `--rest` | 1665.4018e6 | Rest frequency (Hz). Use 1420.405751768e6 for hydrogen |
| `--ra`, `--dec` | W3(OH) | Target position (J2000) |
| `--site` | ATA | Observatory latitude, longitude and height (m) |
| `--start` | from metadata | Override the start time (ISO, UTC) |
| `--center`, `--rate` | from metadata | Override center frequency / sample rate |
| `--no-lsr` | off | Keep velocities topocentric |
| `--out` | `<first file>_spectrum` | Output file prefix |
| `--no-show` | off | Save files without opening a plot window |

---

## 6. What you get

- **A PNG with three panels:**
  1. **Raw averaged spectrum (dB vs frequency offset).** Shows the receiver's bandpass shape, the DC spike at the center and any lines. The dashed line is the estimated baseline.
  2. **Baseline-removed line profile vs LSR velocity**: `(S − baseline) / baseline`, the line power as a fraction of the system noise.
  3. **Dynamic spectrum.** A time-vs-velocity waterfall, 10 s per row.
- **A CSV** with frequency, velocity, raw power and line profile per file, plus the mean, for your own analysis.
- **Console output:** a summary of each file, the LSR correction applied, and the strongest feature in the velocity window with its velocity, frequency, peak and approximate significance (σ).

### Why panel 1 is transformed into panel 2

The power measured in each frequency bin is approximately:

```
measured(f) = gain(f) × [system noise + line(f)]
```

The receiver gain varies across the band and is not known, and the line is a few percent of the system noise. A running median (300 kHz wide) ignores narrow features and estimates `baseline(f) ≈ gain(f) × system noise`. Dividing out the baseline cancels the gain:

```
measured / baseline − 1 = line(f) / system noise
```

The frequency axis is then converted to velocity using the radio Doppler convention, `v = c (f₀ − f) / f₀`. The velocity is then corrected to the local standard of rest (LSRK): the script adds Earth's rotation and orbit (the barycentric correction, from astropy) and the Sun's standard 20 km/s motion. The LSR correction was cross-checked against astropy's independent `SpectralCoord` method and agreed to within 0.003 km/s.

---

## 7. Results and interpretation

LSR correction at the time of the recording: **+14.9 km/s**. Topocentric velocity + 14.9 = LSR velocity.

### Quick test: 60 s, X only, 117 Hz bins

- A single spike at about **−60 km/s topocentric**, about 5% of system power, roughly 4σ.
- The matching bump in panel 1 is at about +335 kHz above 1665.4 MHz, the same frequency.
- This was the first detection. It was weak, but the feature lined up across panels.

### `x_full`: full 506 s, X only, 469 Hz bins, topocentric

- astropy was not installed for this run, so velocities were **topocentric**.
- The console reported a feature at **−59.92 km/s (1665.7347 MHz)**, 0.023 of system power, about **12σ**.
- **Caveat:** the main peak sits at about −60.1 km/s topocentric, just **outside** the `--vmin -60` window. The reported value was the **edge** of the line, not its peak. The lower peak height was not because of the coarser bins. Lesson: the velocity window must match the velocity frame.
- The 4σ → 12σ improvement comes from averaging the full recording instead of 60 s.

### `x_fine`: full 506 s, X only, 117 Hz bins, LSRK

- With astropy installed and fine bins, the maser's structure is resolved:
  - **Main peak at about −45.2 km/s LSR**, about 0.043 of system power.
  - **Second peak at about −46.2 km/s**, about 0.027.
  - A weaker shoulder near −44.5 km/s.
- The emission spans about −46.5 to −44.5 km/s. Each peak is probably a separate maser spot in W3(OH) moving at a slightly different speed. Individual features are about 0.3–0.5 km/s (about 2–3 kHz) wide.
- The velocity is consistent with the roughly −45 km/s commonly cited for W3(OH). That value is from memory and not yet checked against a published spectrum.
- **Dynamic spectrum:** bright horizontal bands at about **200–230 s and 370–470 s** brighten every velocity channel at once. They are not from the source: the total power of the X channel changed, and the script normalizes every row by a single average bandpass. The Y channel does not show these bands (see below), so the most likely cause is a gain jump in the X polarization's own electronics. Polarized interference cannot be fully ruled out.

### `xy_fine`: full 506 s, X + Y, 117 Hz bins, LSRK

- **Panel 1:** the Y channel sits about 2.8 dB above X. That's a gain difference between the two receiver channels, and dividing by each channel's own baseline removes it.
- **Panel 2 (mean of X and Y):**
  - Main peak at **−45.2 km/s**, about 0.049, roughly **15σ** (off-line noise about ±0.003).
  - Second peak at **−46.2 km/s**, about 0.024.
  - Shoulder near **−44.9 km/s**.
  - Small feature near **−44.1 km/s**, about 0.011. It appears in both X and Y, which makes it more likely to be real.
  - The averaged trace is about √2 less noisy than either polarization alone, as expected.
- **X vs Y:**
  - At −46.2 km/s the two polarizations match (about 0.025 each).
  - The main −45.2 km/s peak is stronger in Y (about 0.055) than in X (about 0.043), consistently over several adjacent bins.
  - A difference in system noise between X and Y would scale every feature by the same factor, so a mismatch on only one feature suggests the **−45.2 km/s maser is partly linearly polarized**. This is a 2–3σ difference, so treat it as suggestive, not proven.
- **Panel 3** shows only the first file (X), which is why the X-channel bands appear again.

---

## 8. Caveats and limitations

- **No temperature calibration.** Panel 2 is a ratio, line power / system noise, not kelvins or janskys. To convert, multiply by the system noise expressed as a temperature (a standard way to quantify radio power). That value is not known here for this ATA antenna, so the plot stays as a ratio. Professional observatories get there with calibration against known hot and cold sources.
- **Polarization from X vs Y alone is limited.** Linear feeds see circular polarization equally in X and Y. OH masers are often strongly circularly polarized, so measuring it requires cross-correlating X and Y and computing Stokes parameters. The X–Y phase offset is uncalibrated, which mixes some Stokes parameters.
- **Dynamic spectrum normalization.** Each row is divided by the full-recording bandpass, so gain changes over time show up as horizontal bands. Per-row normalization would remove them.
- **Observatory position** is approximate. An error of a few km changes the velocity correction by far less than 0.01 km/s.
- **Start time.** The recording is assumed to start at 19:36:55 **UTC**. If it were local California time instead, the LSR correction would change by only about 0.1 km/s.
- **Band edges** (outer 6%) and **±1.5 kHz around the DC spike** are excluded from the baseline fit and the line profile.

---

## 9. Possible next steps

- Per-row baseline normalization and an X+Y averaged waterfall.
- A polarization mode: cross-correlate X and Y, then compute Stokes I, Q, U and V.
- Compare the profile against a published W3(OH) 1665 MHz spectrum.
- Reuse the same pipeline on your own RTL-SDR hydrogen-line recordings:
  `--rest 1420.405751768e6` plus your own `--ra`, `--dec` and `--site`.

---

## Source

- *IQ recording of W3OH hydroxyl spectral line at 1665.4 MHz*, Zenodo record 6711078 (Allen Telescope Array, 2022): <https://zenodo.org/records/6711078>
