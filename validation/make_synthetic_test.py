#!/usr/bin/env python3
"""
make_synthetic_test.py - build the synthetic two-polarization test recording
used to validate oh_spectrum.py (per-row normalization and the Stokes mode).

It writes a SigMF pair, PX (X polarization) and PY (Y polarization), in the
same format as the ATA W3(OH) dataset: 16-bit complex integers (ci16_le),
1.92 Msps, centred on 1665.4 MHz. Each file is about 230 MB for 30 s.
The random seed is fixed, so every run produces identical files.

Usage:
    python make_synthetic_test.py                  # writes into the current folder
    python make_synthetic_test.py --out-dir synth  # writes into ./synth

Then, for example:
    python oh_spectrum.py spectrum synth/PX.sigmf-meta synth/PY.sigmf-meta --no-lsr --vmin -48 --vmax -41 --dump 2
    python oh_spectrum.py stokes   synth/PX.sigmf-meta synth/PY.sigmf-meta --no-lsr --vmin -48 --vmax -41

KNOWN PROPERTIES (the "right answers" the analysis should recover)
---------------------------------------------------------------------------
Receiver / instrument
  * Independent noise in X and Y, shaped by a 6th-order low-pass filter
    (flat top, roll-off near the band edges, like a real bandpass).
  * Y channel amplitude gain 1.4x X  ->  20*log10(1.4) = 2.92 dB higher power.
  * Instrumental phase: the Y channel is rotated by +40 deg, so the measured
    cross-power X*conj(Y) carries a -40 deg offset (oh_spectrum.py: --xy-phase -40).
  * Gain jump: X amplitude x1.03 from t = 10 s to 15 s
      -> power change 1.03^2 - 1 = +6.09 % in those rows; Y has no jump.

Five narrow "maser" lines (each ~600 Hz wide, ~30 % of the noise per channel),
at offsets above 1665.4 MHz.  Topocentric radio velocities (rest 1665.4018 MHz):

  label  offset    velocity     polarization            expected Stokes / I
  A      +240 kHz  -42.88 km/s  linear along X          q = +1,  U = V = 0
  E      +245 kHz  -43.78 km/s  linear at +45 deg       u = +1,  q = V = 0
  B      +250 kHz  -44.68 km/s  circular, one hand      v = -1,  q = U = 0
  C      +255 kHz  -45.58 km/s  circular, other hand    v = +1,  q = U = 0
  D      +260 kHz  -46.48 km/s  unpolarized             q = u = v = 0

  With the -40 deg instrumental offset left uncalibrated, the X-Y phases read
  E -40, B -130, C +50 deg: B and C stay 180 deg apart (opposite circular).
  (The sign of V follows this script's convention, V = 2 Im(X conj(Y)).)
"""

import argparse
import json
import os

import numpy as np
from scipy import signal

FS = 1.92e6
CENTER = 1665.4e6
REST = 1665.4018e6
C_KMS = 299792.458
SEED = 7

FEATURES = [  # label, offset (Hz), polarization
    ("A", 240e3, "linear along X"),
    ("E", 245e3, "linear at +45 deg"),
    ("B", 250e3, "circular (V < 0)"),
    ("C", 255e3, "circular (V > 0)"),
    ("D", 260e3, "unpolarized"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=".", help="folder for PX/PY files (default: current)")
    p.add_argument("--seconds", type=float, default=30.0, help="length in seconds (default 30)")
    p.add_argument("--xy-phase", type=float, default=40.0,
                   help="instrumental phase applied to Y, degrees (default 40)")
    p.add_argument("--y-gain", type=float, default=1.4, help="Y amplitude gain (default 1.4)")
    p.add_argument("--jump", type=float, default=1.03,
                   help="X amplitude factor during the gain jump (default 1.03)")
    p.add_argument("--jump-start", type=float, default=10.0, help="gain jump start, s")
    p.add_argument("--jump-end", type=float, default=15.0, help="gain jump end, s")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    n = int(FS * args.seconds)
    step = 1920000                                   # write in 1 s blocks
    rng = np.random.default_rng(SEED)
    phi = np.radians(args.xy_phase)

    b_band, a_band = signal.butter(6, 0.85, "low")           # receiver bandpass
    b_line, a_line = signal.butter(2, 600 / (FS / 2), "low")  # ~600 Hz wide maser line
    state = {k: None for k in ["nx", "ny", "A", "B", "C", "D1", "D2", "E"]}

    def filtered_noise(key, b, a, m):
        """Complex Gaussian noise through a filter, continuous across blocks."""
        w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
        zi = state[key] if state[key] is not None else signal.lfilter_zi(b, a) * 0
        out, state[key] = signal.lfilter(b, a, w, zi=zi)
        return out

    amp = 0.0148 / np.sqrt(2 * 0.00035)              # line ~30 % of noise per channel
    off = {lab: o for lab, o, _ in FEATURES}

    px = os.path.join(args.out_dir, "PX.sigmf-data")
    py = os.path.join(args.out_dir, "PY.sigmf-data")
    with open(px, "wb") as fx, open(py, "wb") as fy:
        for s in range(0, n, step):
            m = min(step, n - s)
            t = (np.arange(m) + s) / FS

            def line(key, offset):
                return amp * filtered_noise(key, b_line, a_line, m) * np.exp(2j * np.pi * offset * t)

            A = line("A", off["A"])
            E = line("E", off["E"])
            B = line("B", off["B"])
            C = line("C", off["C"])
            D1 = line("D1", off["D"])
            D2 = line("D2", off["D"])   # independent copy in Y -> unpolarized

            r2 = np.sqrt(2)
            X = filtered_noise("nx", b_band, a_band, m) + A + E / r2 + B / r2 + C / r2 + D1
            Y_true = filtered_noise("ny", b_band, a_band, m) + E / r2 + 1j * B / r2 - 1j * C / r2 + D2
            Y = args.y_gain * Y_true * np.exp(1j * phi)       # Y-channel gain and phase
            X = X * np.where((t >= args.jump_start) & (t < args.jump_end), args.jump, 1.0)

            for z, f in ((X, fx), (Y, fy)):
                z = z * 2500
                iq = np.empty(2 * m, np.int16)
                iq[0::2] = np.clip(z.real, -32767, 32767)
                iq[1::2] = np.clip(z.imag, -32767, 32767)
                f.write(iq.tobytes())
            print(f"\r  {min(s + step, n) / FS:5.1f} / {args.seconds:g} s", end="", flush=True)
    print()

    # SigMF metadata, with each test feature annotated (inspectrum shows these)
    annotations = [{
        "core:sample_start": 0,
        "core:sample_count": n,
        "core:freq_lower_edge": CENTER + o - 600,
        "core:freq_upper_edge": CENTER + o + 600,
        "core:label": f"{lab}: {pol}",
    } for lab, o, pol in FEATURES]
    for name, desc in (("PX", "X polarization"), ("PY", "Y polarization")):
        meta = {
            "global": {
                "core:datatype": "ci16_le",
                "core:sample_rate": FS,
                "core:version": "1.0.0",
                "core:description": f"Synthetic test recording ({desc}) for oh_spectrum.py; "
                                    f"see make_synthetic_test.py for the known properties.",
            },
            "captures": [{"core:sample_start": 0, "core:frequency": CENTER,
                          "core:datetime": "2022-06-21T19:36:55Z"}],
            "annotations": annotations,
        }
        with open(os.path.join(args.out_dir, name + ".sigmf-meta"), "w") as f:
            json.dump(meta, f, indent=2)

    print(f"Wrote {px} and {py} ({os.path.getsize(px) / 1e6:.0f} MB each) plus .sigmf-meta files")
    print("Injected features (topocentric radio velocity):")
    for lab, o, pol in FEATURES:
        v = C_KMS * (1 - (CENTER + o) / REST)
        print(f"  {lab}  {o / 1e3:+.0f} kHz  {v:7.2f} km/s  {pol}")


if __name__ == "__main__":
    main()
