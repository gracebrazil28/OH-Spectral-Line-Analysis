#!/usr/bin/env python3
"""
oh_spectrum.py - turn a SigMF IQ recording of a radio spectral line into an
averaged, baseline-corrected spectrum with an LSR velocity axis.

Written for the Allen Telescope Array W3(OH) 1665 MHz recording on Zenodo
(https://zenodo.org/records/6711078), but it works on any single-channel SigMF
IQ recording, including your own RTL-SDR hydrogen-line captures
(add  --rest 1420.405751768e6  and your own --ra/--dec/--site).

Pipeline (each step is a function below, so you can read it top to bottom):
  1. read the .sigmf-meta JSON  -> sample rate, center frequency, start time, datatype
  2. memory-map the .sigmf-data -> no need to load 3.9 GB into RAM
  3. chunked, windowed FFTs     -> power spectra, averaged over the whole recording
                                   (plus a coarser "dynamic spectrum" over time)
  4. baseline removal           -> divide out the receiver bandpass with a running median
  5. velocity axis              -> radio-convention Doppler velocity, corrected to the
                                   local standard of rest (LSRK) with astropy
  6. plots + CSV

Requirements:  numpy, scipy, matplotlib   (astropy optional, for the LSR correction)
    conda install -c conda-forge numpy scipy matplotlib astropy

Usage:
    python oh_spectrum.py info      X.sigmf-meta
    python oh_spectrum.py spectrum  X.sigmf-meta Y.sigmf-meta
    python oh_spectrum.py spectrum  X.sigmf-meta --seconds 60          (quick test)
    python oh_spectrum.py stokes    X.sigmf-meta Y.sigmf-meta          (polarization)
    python oh_spectrum.py fit       xy_fine.csv --integration 506.2     (components + distance)
    python oh_spectrum.py wav       X.sigmf-meta                        (WAV for SDR#)

Run any subcommand with -h for all options.
"""

import argparse
import csv
import json
import os
import struct
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np

C_KMS = 299792.458                      # speed of light, km/s

# Defaults for the Zenodo W3(OH) recording ----------------------------------
DEFAULT_REST_HZ = 1665.4018e6           # OH 1665 MHz main line (from the dataset page)
DEFAULT_RA = "02h27m04.1s"              # pointing given on the dataset page (J2000)
DEFAULT_DEC = "+61d52m22s"
# Hat Creek Radio Observatory (Allen Telescope Array). Approximate: an error of
# a few km here changes the velocity correction by far less than 0.01 km/s.
DEFAULT_SITE = (40.8178, -121.4733, 1043.0)   # lat deg, lon deg (east +), height m

# SigMF datatype -> (numpy dtype of one I or Q value, value to subtract)
SIGMF_DTYPES = {
    "ci16_le": ("<i2", 0.0), "ci16_be": (">i2", 0.0), "ci16": ("<i2", 0.0),
    "ci32_le": ("<i4", 0.0), "ci32_be": (">i4", 0.0),
    "cf32_le": ("<f4", 0.0), "cf32_be": (">f4", 0.0), "cf32": ("<f4", 0.0),
    "cf64_le": ("<f8", 0.0), "cf64_be": (">f8", 0.0),
    "ci8": ("i1", 0.0),
    "cu8": ("u1", 127.5),                 # raw RTL-SDR samples
    "cu16_le": ("<u2", 32767.5),
}


# --------------------------------------------------------------------------
# 1. Metadata
# --------------------------------------------------------------------------
def find_pair(path):
    """Accept either half of a SigMF pair and return (meta_path, data_path)."""
    base = path
    for ext in (".sigmf-meta", ".sigmf-data", ".sigmf"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    meta, data = base + ".sigmf-meta", base + ".sigmf-data"
    if not os.path.exists(meta):
        sys.exit(f"Metadata file not found: {meta}")
    if not os.path.exists(data):
        sys.exit(f"Data file not found: {data}\n(Keep the .sigmf-meta and .sigmf-data side by side.)")
    return meta, data


def parse_time(s):
    """Parse a SigMF ISO-8601 timestamp (e.g. 2022-06-24T12:34:56.123456Z)."""
    s = s.strip().replace("Z", "+00:00")
    if "." in s:                         # Python < 3.11 accepts at most 6 fraction digits
        head, frac = s.split(".", 1)
        digits = "".join(ch for ch in frac if ch.isdigit())
        tz = frac[len(digits):]
        s = f"{head}.{digits[:6].ljust(6, '0')}{tz}"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_recording(path, center_override=None, rate_override=None, start_override=None):
    meta_path, data_path = find_pair(path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    g = meta.get("global", {})
    caps = meta.get("captures", [{}]) or [{}]
    cap0 = caps[0]

    datatype = g.get("core:datatype", "")
    if datatype not in SIGMF_DTYPES:
        sys.exit(f"Unsupported or missing core:datatype '{datatype}'. "
                 f"Supported: {', '.join(sorted(SIGMF_DTYPES))}")
    if g.get("core:num_channels", 1) != 1:
        sys.exit("This script expects one channel per file (num_channels = 1).")

    fs = rate_override or g.get("core:sample_rate")
    if not fs:
        sys.exit("No core:sample_rate in metadata; pass --rate.")
    fc = center_override or cap0.get("core:frequency")
    if fc is None:
        sys.exit("No core:frequency in metadata; pass --center (e.g. --center 1665.4e6).")

    start = None
    if start_override:
        start = parse_time(start_override)
    elif cap0.get("core:datetime"):
        start = parse_time(cap0["core:datetime"])

    np_dtype, offset = SIGMF_DTYPES[datatype]
    item = np.dtype(np_dtype).itemsize
    header_bytes = int(cap0.get("core:header_bytes", 0))
    data_bytes = os.path.getsize(data_path) - header_bytes
    n_samples = data_bytes // (2 * item)

    if len(caps) > 1:
        print(f"Note: {len(caps)} capture segments in metadata; using the first one's "
              "frequency and time for the whole file.")

    return {
        "name": os.path.basename(meta_path)[: -len(".sigmf-meta")],
        "meta_path": meta_path, "data_path": data_path, "meta": meta,
        "datatype": datatype, "np_dtype": np_dtype, "offset": offset,
        "header_bytes": header_bytes, "fs": float(fs), "fc": float(fc),
        "start": start, "n_samples": int(n_samples),
        "duration": n_samples / float(fs),
    }


def describe(rec):
    print(f"File          : {rec['data_path']}")
    print(f"Datatype      : {rec['datatype']}")
    print(f"Sample rate   : {rec['fs']:,.0f} samples/s")
    print(f"Center freq   : {rec['fc'] / 1e6:.6f} MHz")
    print(f"Coverage      : {(rec['fc'] - rec['fs'] / 2) / 1e6:.4f} to "
          f"{(rec['fc'] + rec['fs'] / 2) / 1e6:.4f} MHz")
    print(f"Start time    : {rec['start'].isoformat() if rec['start'] else 'not in metadata'}")
    print(f"Samples       : {rec['n_samples']:,}  ({rec['duration']:.1f} s)")


# --------------------------------------------------------------------------
# 2-3. Read in chunks and average FFT power spectra
# --------------------------------------------------------------------------
def integrate(rec, nfft, seconds=None, skip=0.0, dump_seconds=10.0, ffts_per_chunk=64):
    """
    Average |FFT|^2 over the recording.

    Returns
      freqs_off : frequency offset from center for each bin (Hz), fftshifted
      mean_spec : time-averaged power spectrum (arbitrary units), fftshifted
      dyn       : 2-D array [n_dumps, nfft], each row averaged over dump_seconds
      dyn_t     : time (s from start of recording) at the middle of each dump
    """
    fs = rec["fs"]
    raw = np.memmap(rec["data_path"], dtype=rec["np_dtype"], mode="r",
                    offset=rec["header_bytes"])
    first_fft = int(skip * fs) // nfft
    total_ffts = rec["n_samples"] // nfft - first_fft
    if seconds:
        total_ffts = min(total_ffts, int(seconds * fs) // nfft)
    if total_ffts <= 0:
        sys.exit("Nothing to process: check --seconds / --skip against the recording length.")

    ffts_per_dump = max(1, int(round(dump_seconds * fs / nfft)))
    n_dumps = int(np.ceil(total_ffts / ffts_per_dump))

    # Hann window: suppresses spectral leakage so a strong line doesn't
    # smear power into neighbouring bins.
    win = np.hanning(nfft).astype(np.float32)
    acc = np.zeros(nfft)
    dyn = np.zeros((n_dumps, nfft))
    dyn_count = np.zeros(n_dumps)

    t0 = time.time()
    last_report = -1
    done = 0
    while done < total_ffts:
        k = min(ffts_per_chunk, total_ffts - done)
        a = (first_fft + done) * nfft * 2           # index of first I value
        chunk = np.asarray(raw[a: a + 2 * k * nfft], dtype=np.float32)
        if rec["offset"]:
            chunk -= rec["offset"]
        z = chunk.view(np.complex64).reshape(k, nfft)   # interleaved I,Q -> complex
        p = np.abs(np.fft.fft(z * win, axis=1)) ** 2

        acc += p.sum(axis=0)
        # distribute these FFTs into dynamic-spectrum rows
        idx = (done + np.arange(k)) // ffts_per_dump
        for row in np.unique(idx):
            sel = idx == row
            dyn[row] += p[sel].sum(axis=0)
            dyn_count[row] += sel.sum()
        done += k

        pct = int(100 * done / total_ffts)
        if pct // 5 != last_report:
            last_report = pct // 5
            el = time.time() - t0
            eta = el / done * (total_ffts - done)
            print(f"\r  {rec['name']}: {pct:3d}%  ({el:5.0f} s elapsed, ~{eta:4.0f} s left)",
                  end="", flush=True)
    print()

    mean_spec = np.fft.fftshift(acc / total_ffts)
    dyn = np.fft.fftshift(dyn / dyn_count[:, None], axes=1)
    freqs_off = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    dyn_t = skip + (np.arange(n_dumps) + 0.5) * ffts_per_dump * nfft / fs
    # drop a short final row: averaging only a few FFTs makes it noisy and biased
    keep = dyn_count >= 0.5 * ffts_per_dump
    if keep.any():
        dyn, dyn_t = dyn[keep], dyn_t[keep]
    used_s = total_ffts * nfft / fs
    print(f"  integrated {used_s:.1f} s, {total_ffts:,} FFTs, "
          f"resolution {fs / nfft:.1f} Hz/bin")
    return freqs_off, mean_spec, dyn, dyn_t, used_s


def _read_fft_block(raw, rec, start_fft, k, nfft):
    """Read k consecutive FFT-length blocks starting at FFT index start_fft -> complex64 [k, nfft]."""
    a = start_fft * nfft * 2
    chunk = np.asarray(raw[a: a + 2 * k * nfft], dtype=np.float32)
    if rec["offset"]:
        chunk -= rec["offset"]
    return chunk.view(np.complex64).reshape(k, nfft)


def integrate_pair(rx, ry, nfft, seconds=None, skip=0.0, ffts_per_chunk=64):
    """
    Read two simultaneously recorded channels (X and Y) in lockstep and average
      XX  = |Fx|^2          (auto-power of X)
      YY  = |Fy|^2          (auto-power of Y)
      XY  = Fx * conj(Fy)   (cross-power: keeps the relative phase of X and Y)
    The cross-power only averages to something non-zero for signals that are
    present in BOTH channels with a stable phase relation, i.e. polarized emission.
    """
    fs = rx["fs"]
    raw_x = np.memmap(rx["data_path"], dtype=rx["np_dtype"], mode="r", offset=rx["header_bytes"])
    raw_y = np.memmap(ry["data_path"], dtype=ry["np_dtype"], mode="r", offset=ry["header_bytes"])
    first_fft = int(skip * fs) // nfft
    total_ffts = min(rx["n_samples"], ry["n_samples"]) // nfft - first_fft
    if seconds:
        total_ffts = min(total_ffts, int(seconds * fs) // nfft)
    if total_ffts <= 0:
        sys.exit("Nothing to process: check --seconds / --skip against the recording length.")

    win = np.hanning(nfft).astype(np.float32)
    xx = np.zeros(nfft)
    yy = np.zeros(nfft)
    xy = np.zeros(nfft, dtype=np.complex128)
    t0 = time.time()
    last_report = -1
    done = 0
    while done < total_ffts:
        k = min(ffts_per_chunk, total_ffts - done)
        fx = np.fft.fft(_read_fft_block(raw_x, rx, first_fft + done, k, nfft) * win, axis=1)
        fy = np.fft.fft(_read_fft_block(raw_y, ry, first_fft + done, k, nfft) * win, axis=1)
        xx += (fx.real ** 2 + fx.imag ** 2).sum(axis=0)
        yy += (fy.real ** 2 + fy.imag ** 2).sum(axis=0)
        xy += (fx * np.conj(fy)).sum(axis=0)
        done += k
        pct = int(100 * done / total_ffts)
        if pct // 5 != last_report:
            last_report = pct // 5
            el = time.time() - t0
            eta = el / done * (total_ffts - done)
            print(f"\r  X x Y: {pct:3d}%  ({el:5.0f} s elapsed, ~{eta:4.0f} s left)",
                  end="", flush=True)
    print()
    used_s = total_ffts * nfft / fs
    print(f"  integrated {used_s:.1f} s, {total_ffts:,} FFT pairs, "
          f"resolution {fs / nfft:.1f} Hz/bin")
    sh = np.fft.fftshift
    freqs_off = sh(np.fft.fftfreq(nfft, d=1.0 / fs))
    return freqs_off, sh(xx / total_ffts), sh(yy / total_ffts), sh(xy / total_ffts), used_s


# --------------------------------------------------------------------------
# 4. Baseline (receiver bandpass) removal
# --------------------------------------------------------------------------
def bad_bin_mask(freqs_off, fs, edge_frac, dc_hz):
    """True for bins to ignore: band edges (filter roll-off) and the DC spike."""
    half = fs / 2
    mask = np.abs(freqs_off) > half * (1 - edge_frac)
    mask |= np.abs(freqs_off) <= dc_hz
    return mask


def running_median_baseline(spec, bad, window_bins):
    """
    Estimate the smooth bandpass shape with a running median.  A median ignores
    narrow features (like maser lines) as long as the window is much wider than
    the lines, so what's left is the receiver's response.
    """
    from scipy.ndimage import median_filter
    s = spec.copy()
    good = ~bad
    # fill ignored bins by interpolation so they don't distort the median
    s[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), spec[good])
    window_bins = max(3, int(window_bins) | 1)          # odd size
    return median_filter(s, size=window_bins, mode="nearest")


def normalize_rows(dyn, base, bad, mode, window_bins):
    """
    Turn each row of a dynamic spectrum into (S - baseline_row) / baseline_row.

    mode "scale"  : baseline_row = (time-averaged bandpass) x (that row's median gain).
                    Removes broadband gain jumps, which is what causes horizontal bands.
    mode "median" : baseline_row = running median of the row itself.
                    Also follows changes in bandpass SHAPE, but is much slower.
    mode "none"   : baseline_row = time-averaged bandpass (the old behaviour).

    Returns (normalized rows, relative gain of each row).
    """
    good = ~bad
    ratio = dyn / base[None, :]
    gains = np.median(ratio[:, good], axis=1)
    if mode == "none":
        norm = ratio - 1.0
    elif mode == "scale":
        norm = ratio / gains[:, None] - 1.0
    elif mode == "median":
        norm = np.empty_like(dyn)
        for i, row in enumerate(dyn):
            norm[i] = row / running_median_baseline(row, bad, window_bins) - 1.0
    else:
        sys.exit(f"Unknown --row-norm mode '{mode}'")
    norm[:, bad] = np.nan
    return norm, gains / np.median(gains)


# --------------------------------------------------------------------------
# 5. Velocity axis
# --------------------------------------------------------------------------
def lsr_correction_kms(ra, dec, when_utc, site):
    """
    Velocity (km/s) to ADD to a topocentric velocity to get an LSRK velocity.
      = barycentric correction (Earth's spin + orbit, from astropy)
      + Sun's motion relative to the LSR (standard 20 km/s toward RA 18h, Dec +30, B1900)
    Returns None if astropy is not installed.
    """
    try:
        import astropy.units as u
        from astropy.coordinates import FK4, EarthLocation, SkyCoord
        from astropy.time import Time
    except ImportError:
        return None
    lat, lon, h = site
    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=h * u.m)
    t = Time(when_utc.astimezone(timezone.utc).replace(tzinfo=None), scale="utc")
    tgt = SkyCoord(ra, dec, frame="icrs")
    v_bary = tgt.radial_velocity_correction(kind="barycentric", obstime=t,
                                            location=loc).to(u.km / u.s).value
    apex = SkyCoord(ra=18 * u.hourangle, dec=30 * u.deg,
                    frame=FK4(equinox="B1900")).icrs
    v_sun = 20.0 * np.cos(tgt.separation(apex).rad)
    return v_bary + v_sun


def radio_velocity_kms(freq_hz, rest_hz):
    """Radio-convention Doppler velocity: v = c (f0 - f) / f0. Positive = receding."""
    return C_KMS * (1.0 - freq_hz / rest_hz)


def velocity_axis(rec, args, f_rf, used_s):
    """Velocity for each channel, LSRK-corrected when possible. Returns (v, axis label)."""
    v_topo = radio_velocity_kms(f_rf, args.rest)
    mid_time = None
    if rec["start"] is not None:
        mid_time = rec["start"] + timedelta(seconds=args.skip + used_s / 2)
    corr = None
    if args.no_lsr:
        print("LSR correction disabled (--no-lsr): velocities are topocentric.")
    elif mid_time is None:
        print("No start time in metadata (pass --start): velocities are topocentric.")
    else:
        corr = lsr_correction_kms(args.ra, args.dec, mid_time, args.site)
        if corr is None:
            print("astropy not installed: velocities are topocentric "
                  "(conda install -c conda-forge astropy to enable LSR).")
    if corr is not None:
        print(f"LSRK correction at {mid_time.isoformat()} : {corr:+.3f} km/s")
        return v_topo + corr, "LSRK velocity (km/s, radio)"
    return v_topo, "Topocentric velocity (km/s, radio)"


# --------------------------------------------------------------------------
# 6. Spectrum command: process, plot, save
# --------------------------------------------------------------------------
def cmd_spectrum(args):
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [load_recording(p, args.center, args.rate, args.start) for p in args.files]
    fs, fc = recs[0]["fs"], recs[0]["fc"]
    for r in recs[1:]:
        if r["fs"] != fs or r["fc"] != fc:
            sys.exit("All files must have the same sample rate and center frequency.")
    for r in recs:
        describe(r)
        print()

    nfft = args.nfft
    print(f"FFT size {nfft} -> {fs / nfft:.1f} Hz per bin "
          f"({C_KMS * fs / nfft / args.rest * 1000:.1f} m/s per bin at the rest frequency)")

    results = []
    for r in recs:
        f_off, spec, dyn, dyn_t, used_s = integrate(
            r, nfft, seconds=args.seconds, skip=args.skip, dump_seconds=args.dump)
        results.append((r, spec, dyn, dyn_t, used_s))

    # frequency & velocity axes
    f_rf = fc + f_off
    v, vlabel = velocity_axis(recs[0], args, f_rf, results[0][4])

    # baseline-normalize each polarization: (S - B) / B  = line / system power
    bad = bad_bin_mask(f_off, fs, args.edge, args.dc_mask)
    win_bins = args.baseline_khz * 1e3 / (fs / nfft)
    lines, baselines = [], []
    for r, spec, dyn, dyn_t, _ in results:
        base = running_median_baseline(spec, bad, win_bins)
        line = spec / base - 1.0
        line[bad] = np.nan
        lines.append(line)
        baselines.append(base)
    # masked bins are NaN in every file, so they stay NaN in the mean
    total = np.vstack(lines).mean(axis=0) if len(lines) > 1 else lines[0]

    # report the strongest peak inside the plotted velocity window
    sel = (v >= args.vmin) & (v <= args.vmax) & ~bad
    if sel.any():
        i = np.flatnonzero(sel)[np.nanargmax(total[sel])]
        noise = 1.4826 * np.nanmedian(np.abs(total[~bad] - np.nanmedian(total[~bad])))
        print(f"Strongest feature in window: {v[i]:+.2f} km/s  "
              f"({f_rf[i] / 1e6:.6f} MHz), peak {total[i]:.4f} of system power, "
              f"~{total[i] / noise:.0f} sigma")

    # ---------- dynamic spectra: normalize each row by its own baseline ----------
    dyn_norms, row_gains = [], []
    for (r, spec, dyn, dyn_t, _), base in zip(results, baselines):
        dn, g = normalize_rows(dyn, base, bad, args.row_norm, win_bins)
        dyn_norms.append(dn)
        row_gains.append(g)
        dev = (g - 1) * 100
        print(f"  {r['name']}: row gain varies {dev.min():+.2f}% to {dev.max():+.2f}% "
              f"(largest at t = {dyn_t[np.argmax(np.abs(dev))]:.0f} s)")
    n_rows = min(d.shape[0] for d in dyn_norms)
    dyn_mean = np.mean([d[:n_rows] for d in dyn_norms], axis=0)
    dyn_t0 = results[0][3][:n_rows]

    # ---------- plots ----------
    fig, axes = plt.subplots(4, 1, figsize=(11, 15), constrained_layout=True,
                             gridspec_kw={"height_ratios": [3, 3, 3, 1.4]})
    ax = axes[0]
    for (r, spec, *_), base in zip(results, baselines):
        ax.plot(f_off / 1e3, 10 * np.log10(spec), lw=0.6, label=f"{r['name']} raw")
        ax.plot(f_off / 1e3, 10 * np.log10(base), lw=1.2, ls="--", label=f"{r['name']} baseline")
    ax.set_xlabel(f"Offset from {fc / 1e6:.4f} MHz (kHz)")
    ax.set_ylabel("Power (dB, arbitrary)")
    ax.set_title("1. Averaged spectrum: receiver bandpass with the line(s) on top")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    order = np.argsort(v)
    if len(lines) > 1:
        for (r, *_), line in zip(results, lines):
            ax.plot(v[order], line[order], lw=0.6, alpha=0.6, label=r["name"])
    ax.plot(v[order], total[order], lw=1.0, color="k",
            label="mean of files" if len(lines) > 1 else results[0][0]["name"])
    ax.set_xlim(args.vmin, args.vmax)
    in_win = total[sel] if sel.any() else total[~bad]
    lo, hi = np.nanmin(in_win), np.nanmax(in_win)
    pad = 0.08 * (hi - lo if hi > lo else 1)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(vlabel)
    ax.set_ylabel("(S - baseline) / baseline")
    ax.set_title("2. Baseline-removed line profile")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    cols = np.flatnonzero(sel) if sel.any() else np.flatnonzero(~bad)
    cols = cols[np.argsort(v[cols])]
    img = dyn_mean[:, cols]
    vmin_c, vmax_c = np.nanpercentile(img, [1, 99.5])
    ax.imshow(img, aspect="auto", origin="lower", cmap="viridis",
              vmin=vmin_c, vmax=vmax_c,
              extent=[v[cols][0], v[cols][-1], dyn_t0[0] - args.dump / 2,
                      dyn_t0[-1] + args.dump / 2])
    ax.set_xlabel(vlabel)
    ax.set_ylabel("Time from start (s)")
    who = "mean of " + " + ".join(r["name"].rsplit("_", 1)[-1] for r in recs) \
        if len(recs) > 1 else recs[0]["name"]
    norm_txt = {"scale": "each row rescaled to its own gain",
                "median": "each row divided by its own running median",
                "none": "single time-averaged baseline"}[args.row_norm]
    ax.set_title(f"3. Dynamic spectrum ({who}; {args.dump:g} s rows; {norm_txt})")

    ax = axes[3]
    for (r, *_), g in zip(results, row_gains):
        ax.plot(results[0][3][:len(g)], (g - 1) * 100, marker=".", lw=1, label=r["name"])
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Time from start (s)")
    ax.set_ylabel("Gain change (%)")
    ax.set_title(f"4. Broadband gain of each {args.dump:g} s row relative to the median "
                 "(this is what per-row normalization removes)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    out_base = args.out or recs[0]["name"] + "_spectrum"
    png = out_base + ".png"
    fig.suptitle(f"Rest {args.rest / 1e6:.4f} MHz | {sum(x[4] for x in results) / len(results):.0f} s "
                 f"integrated | {fs / nfft:.0f} Hz/bin", fontsize=10)
    fig.savefig(png, dpi=130)
    print(f"Saved plot : {png}")

    csv_path = out_base + ".csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["freq_hz", "velocity_kms"]
        for r, *_ in results:
            hdr += [f"{r['name']}_power", f"{r['name']}_line"]
        hdr += ["mean_line"]
        w.writerow(hdr)
        for i in range(nfft):
            row = [f"{f_rf[i]:.3f}", f"{v[i]:.4f}"]
            for (r, spec, *_), line in zip(results, lines):
                row += [f"{spec[i]:.6e}", "" if np.isnan(line[i]) else f"{line[i]:.6e}"]
            row.append("" if np.isnan(total[i]) else f"{total[i]:.6e}")
            w.writerow(row)
    print(f"Saved data : {csv_path}")

    if not args.no_show:
        plt.show()


# --------------------------------------------------------------------------
# 7. Stokes command: polarization from the X-Y cross-correlation
# --------------------------------------------------------------------------
# For linear feeds X and Y, the four Stokes parameters come from the two
# auto-powers and the cross-power:
#     I = XX + YY            total intensity
#     Q = XX - YY            linear polarization along X (+) or Y (-)
#     U = 2 Re(X Y*)         linear polarization at +/-45 degrees
#     V = 2 Im(X Y*)         circular polarization (sign is convention-dependent)
# Each auto-power is divided by its own system-noise baseline (removing the
# 2.8 dB X/Y gain difference) and the cross-power by sqrt(Bx * By), so every
# Stokes parameter is in units of "fraction of one channel's system noise".
#
# Calibration caveat: the electronics behind X and Y add an unknown phase
# offset phi to X Y*. That rotates (U, V) into (U cos phi - V sin phi, ...),
# so without a calibrator U and V cannot be separated. Two things survive:
#   * sqrt(U^2 + V^2) = 2 |X Y*|           (phi drops out of the magnitude)
#   * phase DIFFERENCES between features   (phi is the same for all of them)
# Q also needs no phase calibration, but assumes X and Y have equal system
# temperatures.

def robust_std(a):
    a = np.asarray(a)
    a = a[np.isfinite(a)]
    return 1.4826 * np.median(np.abs(a - np.median(a)))


def block_sum_sigma(arr, mask, n):
    """
    Noise of a sum over n adjacent channels, measured empirically from
    line-free channels. This includes the correlation between neighbouring
    channels (from the Hann window) that sigma * sqrt(n) would miss.
    """
    idx = np.flatnonzero(mask)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    sums = [arr[run[: (len(run) // n) * n]].reshape(-1, n).sum(axis=1)
            for run in runs if len(run) >= n]
    sums = np.concatenate(sums) if sums else np.array([])
    if sums.size < 20:                       # not enough line-free data: fall back
        return robust_std(arr[mask]) * np.sqrt(n)
    return robust_std(sums)


def find_peaks_simple(a, height, distance):
    """
    Indices of local maxima above `height`, at least `distance` channels apart
    (strongest kept first). Plain-numpy stand-in for scipy.signal.find_peaks.
    """
    a = np.asarray(a, dtype=float)
    cand = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:]) & (a[1:-1] > height)) + 1
    chosen = []
    for i in cand[np.argsort(a[cand])[::-1]]:
        if all(abs(i - j) >= distance for j in chosen):
            chosen.append(i)
    return np.array(sorted(chosen), dtype=int)


def cmd_stokes(args):
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rx = load_recording(args.xfile, args.center, args.rate, args.start)
    ry = load_recording(args.yfile, args.center, args.rate, args.start)
    for r in (rx, ry):
        describe(r)
        print()
    if rx["fs"] != ry["fs"] or rx["fc"] != ry["fc"]:
        sys.exit("X and Y must have the same sample rate and center frequency.")
    if rx["data_path"] == ry["data_path"]:
        sys.exit("X and Y are the same file. Pass the pol_x file first, then pol_y.")
    if rx["start"] and ry["start"]:
        dt = abs((rx["start"] - ry["start"]).total_seconds())
        if dt > 1e-3:
            print(f"WARNING: X and Y start times differ by {dt:.6f} s. The cross-correlation "
                  "assumes the files are sample-aligned; an offset will weaken it.")
    if rx["n_samples"] != ry["n_samples"]:
        print(f"Note: X has {rx['n_samples']:,} samples and Y has {ry['n_samples']:,}; "
              "using the shorter length.")

    fs, fc, nfft = rx["fs"], rx["fc"], args.nfft
    print(f"FFT size {nfft} -> {fs / nfft:.1f} Hz per bin")
    f_off, XX, YY, XY, used_s = integrate_pair(rx, ry, nfft, args.seconds, args.skip)
    f_rf = fc + f_off
    v, vlabel = velocity_axis(rx, args, f_rf, used_s)

    # ---- baselines and normalization ----
    bad = bad_bin_mask(f_off, fs, args.edge, args.dc_mask)
    good = ~bad
    win_bins = args.baseline_khz * 1e3 / (fs / nfft)
    Bx = running_median_baseline(XX, bad, win_bins)
    By = running_median_baseline(YY, bad, win_bins)
    # Off the line, X Y* should average to ~0. Anything left is correlated
    # receiver noise or crosstalk; remove it like a bandpass.
    Bxy = (running_median_baseline(XY.real, bad, win_bins)
           + 1j * running_median_baseline(XY.imag, bad, win_bins))
    norm = np.sqrt(Bx * By)

    x = XX / Bx - 1.0
    y = YY / By - 1.0
    c = (XY - Bxy) / norm
    calibrated = args.xy_phase is not None
    if calibrated:
        c = c * np.exp(-1j * np.radians(args.xy_phase))

    I, Q = x + y, x - y
    U, V = 2 * c.real, 2 * c.imag
    P = 2 * np.abs(c)                        # sqrt(U^2 + V^2), independent of the phase offset
    for a in (I, Q, U, V, P):
        a[bad] = np.nan
    c[bad] = np.nan

    floor = np.abs(np.median(Bxy[good].real) + 1j * np.median(Bxy[good].imag)) / np.median(norm[good])
    print(f"Correlated X-Y noise off the line (removed as a baseline): {floor:.4f} of system power")

    win = good & (v >= args.vmin) & (v <= args.vmax)
    off = good & ~((v >= args.vmin) & (v <= args.vmax))
    sI, sQ, sU, sV = (robust_std(a[off]) for a in (I, Q, U, V))
    sP = 0.5 * (sU + sV)
    print(f"Per-channel noise off the line: I {sI:.4f}, Q {sQ:.4f}, U' {sU:.4f}, V' {sV:.4f}")

    # ---- find features ----
    chan_kms = abs(v[1] - v[0])
    if args.features:
        centers = sorted(args.features)
    else:
        sm = np.convolve(np.nan_to_num(I), np.ones(5) / 5, mode="same")
        s_sm = robust_std(sm[off])
        dist = max(1, int(round(2 * args.halfwidth / chan_kms)))
        pk = find_peaks_simple(np.where(win, sm, 0.0), args.min_snr * s_sm, dist)
        centers = sorted(v[pk])
        print(f"Automatic peak finding (> {args.min_snr:g} sigma, 5-channel smoothing): "
              f"{len(centers)} feature(s)")

    # ---- measure each feature ----
    feats = []
    for vc in centers:
        m = good & (np.abs(v - vc) <= args.halfwidth)
        n = int(m.sum())
        if n == 0:
            continue
        Is, Qs = np.nansum(I[m]), np.nansum(Q[m])
        cs = np.nansum(c[m])
        Us, Vs, Ps = 2 * cs.real, 2 * cs.imag, 2 * abs(cs)
        eI = block_sum_sigma(I, off, n)
        eQ = block_sum_sigma(Q, off, n)
        eU = block_sum_sigma(U, off, n)
        eV = block_sum_sigma(V, off, n)
        eP = 0.5 * (eU + eV)
        P_deb = np.sqrt(max(Ps ** 2 - eP ** 2, 0.0))   # remove the positive noise bias of |.|

        def frac(num, enum):
            """num / I with an error that includes the noise in I as well."""
            r = num / Is
            return r, np.sqrt(enum ** 2 + (r * eI) ** 2) / abs(Is)

        q, eq = frac(Qs, eQ)
        u, eu = frac(Us, eU)
        vv, ev = frac(Vs, eV)
        p, ep = frac(P_deb, eP)
        feats.append(dict(
            v=vc, n=n, snr=Is / eI,
            q=q, eq=eq, u=u, eu=eu, vv=vv, ev=ev,
            p=p, ep=ep, p_snr=Ps / eP,
            phase=np.degrees(np.angle(cs)),
            ephase=np.degrees(eP / Ps) if Ps > 0 else np.nan,
        ))

    # ---- report ----
    up = "U " if calibrated else "U'"
    vp = "V " if calibrated else "V'"
    print()
    print(f"Feature measurements (window +/- {args.halfwidth:g} km/s; fractions of Stokes I):")
    hdr = (f"{'v (km/s)':>9} {'I S/N':>6} | {'q = Q/I':>14} | "
           f"{'sqrt(U2+V2)/I':>14} | {'X-Y phase':>13}")
    if calibrated:
        hdr += f" | {'u = U/I':>14} | {'v = V/I':>14}"
    print(hdr)
    print("-" * len(hdr))
    for f in feats:
        line = (f"{f['v']:>9.2f} {f['snr']:>6.0f} | {f['q']:+.3f} +/- {f['eq']:.3f} | "
                f"{f['p']:.3f} +/- {f['ep']:.3f} | ")
        line += (f"{f['phase']:+6.0f} +/- {f['ephase']:3.0f}" if f["p_snr"] > 3
                 else f"{'(not detected)':>13}")
        if calibrated:
            line += f" | {f['u']:+.3f} +/- {f['eu']:.3f} | {f['vv']:+.3f} +/- {f['ev']:.3f}"
        print(line)
    print()
    print("How to read this:")
    print("  q            : X vs Y difference. Non-zero at > 3 sigma -> linear polarization along X or Y")
    print("                 (assumes equal system temperatures in X and Y).")
    print("  sqrt(U2+V2)/I: polarized signal that is correlated between X and Y: linear at +/-45 deg")
    print("                 and/or circular. Not affected by the unknown X-Y phase.")
    if not calibrated:
        print("  X-Y phase    : includes an unknown instrumental offset, so only DIFFERENCES between")
        print("                 features mean something. Features ~180 deg apart have opposite-sign")
        print("                 U or V; for OH masers, opposite circular polarization is the signature")
        print("                 of a Zeeman pair.")

    # ---- plots ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True, sharex=True)
    order = np.argsort(v)
    vo = v[order]

    def shade(ax):
        for f in feats:
            ax.axvspan(f["v"] - args.halfwidth, f["v"] + args.halfwidth, color="0.85", zorder=0)

    ax = axes[0]
    shade(ax)
    ax.plot(vo, I[order], color="k", lw=1.0, label="I = X + Y")
    ax.plot(vo, Q[order], color="tab:red", lw=0.8, label="Q = X - Y")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_ylabel("fraction of system noise")
    ax.set_title("1. Total intensity I and linear polarization Q (X vs Y)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    shade(ax)
    ax.plot(vo, U[order], color="tab:blue", lw=0.8,
            label="U = 2 Re(XY*)" if calibrated else "U' = 2 Re(XY*)  (mixed with V)")
    ax.plot(vo, V[order], color="tab:green", lw=0.8,
            label="V = 2 Im(XY*)" if calibrated else "V' = 2 Im(XY*)  (mixed with U)")
    ax.plot(vo, P[order], color="k", lw=1.0, label="2|XY*| = sqrt(U² + V²)")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_ylabel("fraction of system noise")
    ax.set_title("2. Cross-correlation of X and Y: polarized emission common to both feeds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    shade(ax)
    det = (P > 4 * sP) & np.isfinite(P)
    ax.scatter(v[det], np.degrees(np.angle(c[det])), s=8, c="tab:purple",
               label=f"channels with 2|XY*| > 4 sigma")
    for f in feats:
        if f["p_snr"] > 3:
            ax.errorbar(f["v"], f["phase"], yerr=f["ephase"], fmt="o", color="k", ms=6)
    ax.set_ylim(-190, 190)
    ax.set_yticks(range(-180, 181, 90))
    ax.set_ylabel("phase of XY* (deg)")
    ax.set_xlabel(vlabel)
    ax.set_title("3. X-Y phase" + ("" if calibrated else
                 " (includes unknown instrumental offset; compare features, not absolute values)"))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xlim(args.vmin, args.vmax)

    fig.suptitle(f"Polarization: {rx['name']} x {ry['name']} | {used_s:.0f} s | "
                 f"{fs / nfft:.0f} Hz/bin | X-Y phase "
                 + (f"calibrated ({args.xy_phase:+g} deg)" if calibrated else "NOT calibrated"),
                 fontsize=10)
    out_base = args.out or rx["name"] + "_stokes"
    png = out_base + ".png"
    fig.savefig(png, dpi=130)
    print(f"\nSaved plot : {png}")

    csv_path = out_base + ".csv"
    with open(csv_path, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["freq_hz", "velocity_kms", "I", "Q", "U" if calibrated else "U_prime",
                    "V" if calibrated else "V_prime", "sqrt_U2_V2"])
        for i in range(nfft):
            vals = [I[i], Q[i], U[i], V[i], P[i]]
            w.writerow([f"{f_rf[i]:.3f}", f"{v[i]:.4f}"]
                       + ["" if np.isnan(a) else f"{a:.6e}" for a in vals])
    print(f"Saved data : {csv_path}")

    if not args.no_show:
        plt.show()


# --------------------------------------------------------------------------
# 8. Fit command: Gaussian components + kinematic distance, from a spectrum CSV
# --------------------------------------------------------------------------
FIT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FIT_INK, FIT_INK2, FIT_GRID, FIT_SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))      # 2.3548
AMU_KG, K_B = 1.66054e-27, 1.380649e-23


def gauss_sum(v, *p):
    """Sum of Gaussians; p = (amplitude, center, sigma) repeated."""
    out = np.zeros_like(v, dtype=float)
    for a, c, w in zip(p[0::3], p[1::3], p[2::3]):
        out += a * np.exp(-0.5 * ((v - c) / w) ** 2)
    return out


def fit_gaussians(v, y, sig, guesses, args):
    """Least-squares fit of len(guesses) Gaussians. Returns dict or None if it fails."""
    from scipy.optimize import curve_fit
    w_lo, w_hi = args.min_fwhm / FWHM_PER_SIGMA, args.max_fwhm / FWHM_PER_SIGMA
    a_hi = 2.0 * np.max(y) + 10 * sig
    p0, lo, hi = [], [], []
    for g in guesses:
        a0 = float(np.clip(np.interp(g, v, y), sig, a_hi * 0.9))
        p0 += [a0, g, float(np.clip(0.15, w_lo * 1.01, w_hi * 0.99))]
        lo += [0.0, g - args.center_range, w_lo]
        hi += [a_hi, g + args.center_range, w_hi]
    try:
        p, cov = curve_fit(gauss_sum, v, y, p0=p0, bounds=(lo, hi),
                           sigma=np.full(v.size, sig), absolute_sigma=True, maxfev=50000)
    except (RuntimeError, ValueError) as exc:
        print(f"  fit with {len(guesses)} components failed: {exc}")
        return None
    resid = y - gauss_sum(v, *p)
    chi2 = float(np.sum((resid / sig) ** 2))
    k, n = len(p), v.size
    return {"p": p, "err": np.sqrt(np.diag(cov)), "chi2": chi2, "red_chi2": chi2 / max(n - k, 1),
            "bic": chi2 + k * np.log(n), "guesses": list(guesses), "resid": resid}


def kinematic_curve(l_rad, b_rad, r0, theta0, dmax=20.0, n=4000):
    """LSR velocity vs distance for a flat rotation curve along (l, b)."""
    d = np.linspace(0.01, dmax, n)
    dp = d * np.cos(b_rad)                               # distance projected on the plane
    R = np.sqrt(r0 ** 2 + dp ** 2 - 2 * r0 * dp * np.cos(l_rad))
    v = (theta0 * r0 / R - theta0) * np.sin(l_rad) * np.cos(b_rad)
    return d, v


def cmd_fit(args):
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- read the spectrum CSV written by the 'spectrum' command ----
    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        rows = list(reader)
    line_cols = [c for c in cols if c.endswith("_line") and c != "mean_line"]
    column = args.column or ("mean_line" if "mean_line" in cols else (line_cols[0] if line_cols else None))
    if column not in cols or "velocity_kms" not in cols:
        sys.exit(f"CSV needs 'velocity_kms' and '{column}' columns. Columns found: {', '.join(cols)}")
    n_pol = len(line_cols) if column == "mean_line" else 1

    v, y, fr = [], [], []
    for r in rows:
        if r[column] == "":
            continue                                      # masked channel (band edge / DC)
        v.append(float(r["velocity_kms"]))
        y.append(float(r[column]))
        fr.append(float(r["freq_hz"]))
    v, y, fr = np.array(v), np.array(y), np.array(fr)
    o = np.argsort(v)
    v, y, fr = v[o], y[o], fr[o]
    chan_hz = float(np.median(np.abs(np.diff(np.sort(fr)))))

    # ---- noise from line-free velocity ranges ----
    if len(args.noise) % 2:
        sys.exit("--noise takes pairs of velocities: VMIN1 VMAX1 [VMIN2 VMAX2 ...]")
    noise_mask = np.zeros(v.size, bool)
    for a, b in zip(args.noise[0::2], args.noise[1::2]):
        noise_mask |= (v >= min(a, b)) & (v <= max(a, b))
    if noise_mask.sum() < 50:
        sys.exit("Too few channels in the --noise ranges; widen them.")
    sig = robust_std(y[noise_mask])
    win = (v >= args.vmin) & (v <= args.vmax)
    if win.sum() < 10:
        sys.exit("Too few channels in the fit window; check --vmin/--vmax and the velocity frame.")
    vf, yf = v[win], y[win]

    print(f"Profile column : {column}  ({n_pol} polarization{'s' if n_pol > 1 else ''} averaged)")
    print(f"Channel width  : {chan_hz:.1f} Hz")
    print(f"Noise (measured, line-free channels): {sig:.5f} per channel")
    if args.integration:
        pred = 1.0 / np.sqrt(chan_hz * args.integration * n_pol)
        print(f"Noise (radiometer equation, {args.integration:g} s x {n_pol} pol): {pred:.5f}  "
              f"-> measured / predicted = {sig / pred:.2f}")

    # ---- fit: start from the guesses, then add components while the BIC improves ----
    print(f"\nFitting {args.vmin:g} to {args.vmax:g} km/s ...")
    tried = []
    best = fit_gaussians(vf, yf, sig, args.guess, args)
    if best is None:
        sys.exit("The starting fit failed; try different --guess values.")
    tried.append(best)
    for _ in range(args.max_extra):
        sm = np.convolve(best["resid"], np.ones(5) / 5, mode="same")
        new_v = float(vf[np.argmax(sm)])                  # biggest positive residual
        cand = fit_gaussians(vf, yf, sig, best["guesses"] + [new_v], args)
        if cand is None:
            break
        tried.append(cand)
        if cand["bic"] < best["bic"]:
            best = cand
        else:
            break
    print(f"{'components':>11} {'reduced chi2':>13} {'BIC':>9}")
    for t in tried:
        mark = "  <- chosen (lowest BIC)" if t is best else ""
        print(f"{len(t['guesses']):>11} {t['red_chi2']:>13.2f} {t['bic']:>9.1f}{mark}")

    # ---- component table ----
    p, e = best["p"], best["err"]
    comps = sorted([(p[i], p[i + 1], p[i + 2], e[i], e[i + 1], e[i + 2])
                    for i in range(0, len(p), 3)], key=lambda c: c[1])
    print("\nComponents (uncertainties assume independent channels; the Hann window makes "
          "neighbouring channels correlated, so treat them as lower limits):")
    print(f"{'v (km/s)':>17} {'peak':>17} {'FWHM (km/s)':>17} {'area (km/s)':>12}")
    table = []
    for a, c, w, ea, ec, ew in comps:
        fwhm, efwhm = FWHM_PER_SIGMA * w, FWHM_PER_SIGMA * ew
        area = a * w * np.sqrt(2 * np.pi)
        tag = "  (broad)" if fwhm > 1.0 else ""
        if fwhm <= args.min_fwhm * 1.02 or fwhm >= args.max_fwhm * 0.98:
            tag += "  (width at --min/--max-fwhm limit: poorly constrained)"
        print(f"{c:>9.3f} +/- {ec:.3f} {a:>9.4f} +/- {ea:.4f} {fwhm:>9.3f} +/- {efwhm:.3f} "
              f"{area:>12.5f}{tag}")
        table.append((c, ec, a, ea, fwhm, efwhm, area))

    mol = args.mass_amu * AMU_KG
    print(f"\nThermal FWHM for a molecule of {args.mass_amu:g} amu "
          "(lines narrower than this point to maser amplification):")
    print("  " + ", ".join(f"{T:g} K: {FWHM_PER_SIGMA * np.sqrt(K_B * T / mol) / 1e3:.2f} km/s"
                           for T in args.tkin))

    out_base = args.out or os.path.splitext(os.path.basename(args.csv))[0] + "_fit"
    with open(out_base + ".csv", "w", newline="") as f:
        w_ = csv.writer(f)
        w_.writerow(["velocity_kms", "velocity_err", "peak", "peak_err",
                     "fwhm_kms", "fwhm_err", "area_kms"])
        for row in table:
            w_.writerow([f"{x:.6g}" for x in row])
    print(f"\nSaved table: {out_base}.csv")

    # ---- style shared by both figures ----
    style = {"figure.facecolor": FIT_SURF, "axes.facecolor": FIT_SURF,
             "axes.edgecolor": FIT_INK2, "axes.labelcolor": FIT_INK, "xtick.color": FIT_INK2,
             "ytick.color": FIT_INK2, "text.color": FIT_INK, "axes.grid": True,
             "grid.color": FIT_GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
             "axes.spines.right": False, "font.size": 10}

    with plt.rc_context(style):
        # ---- figure 1: components ----
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True,
                                      gridspec_kw={"height_ratios": [3, 1]})
        ax.plot(vf, yf, color="#9a9893", lw=0.8, label=f"measured ({column})")
        vv = np.linspace(args.vmin, args.vmax, 3000)
        total = gauss_sum(vv, *p)
        for k, (a, c, w, *_rest) in enumerate(comps):
            fwhm = FWHM_PER_SIGMA * w
            name = "broad wing" if fwhm > 1.0 else f"{c:.2f} km/s"
            ax.plot(vv, a * np.exp(-0.5 * ((vv - c) / w) ** 2),
                    color=FIT_COLORS[k % len(FIT_COLORS)], lw=2,
                    label=f"{name}: FWHM {fwhm:.2f} km/s")
            if fwhm <= 1.0:
                ax.annotate(f"{c:.2f}", (c, gauss_sum(np.array([c]), *p)[0]), xytext=(0, 10),
                            textcoords="offset points", ha="center", color=FIT_INK, fontsize=9)
        ax.plot(vv, total, color=FIT_INK, lw=1.4, label="sum of components")
        ax.set_ylim(top=max(np.max(yf), np.max(total)) * 1.15)
        ax.set_ylabel("line / system noise")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        ax.set_title(f"Gaussian decomposition ({len(comps)} components, reduced "
                     f"chi2 {best['red_chi2']:.2f})", loc="left", fontsize=11)
        axr.plot(vf, best["resid"] / sig, color=FIT_INK2, lw=0.7)
        axr.axhspan(-2, 2, color=FIT_GRID, zorder=0)
        axr.axhline(0, color=FIT_INK2, lw=0.6)
        axr.set_ylim(-5, 5)
        axr.set_ylabel("residual (sigma)")
        axr.set_xlabel("velocity (km/s), same frame as the CSV")
        fig.savefig(out_base + ".png", dpi=140)
        print(f"Saved plot : {out_base}.png")

        # ---- figure 2: kinematic distance ----
        if not args.no_kinematic:
            try:
                from astropy.coordinates import SkyCoord
            except ImportError:
                print("astropy not installed: skipping the kinematic-distance plot.")
                SkyCoord = None
            if SkyCoord is not None:
                gal = SkyCoord(args.ra, args.dec, frame="icrs").galactic
                l, b = gal.l.rad, gal.b.rad
                v_src = args.kin_velocity if args.kin_velocity is not None else \
                    max(comps, key=lambda c: c[0])[1]        # strongest component
                d, vr = kinematic_curve(l, b, args.r0, args.theta0)
                cross = np.flatnonzero(np.diff(np.sign(vr - v_src)) != 0)
                d_kin = [float(np.interp(v_src, sorted(vr[i:i + 2]),
                                         d[i:i + 2] if vr[i] < vr[i + 1] else d[i:i + 2][::-1]))
                         for i in cross]
                print(f"\nKinematic distance for {v_src:.2f} km/s at l = {np.degrees(l):.2f}, "
                      f"b = {np.degrees(b):.2f} deg (flat curve, R0 {args.r0:g} kpc, "
                      f"Theta0 {args.theta0:g} km/s): "
                      + (", ".join(f"{x:.2f} kpc" for x in d_kin) if d_kin
                         else "none (velocity not allowed by circular rotation here)"))
                xmax = max([6.0] + [x * 1.3 for x in d_kin]
                           + ([args.parallax * 1.5] if args.parallax else []))
                sel = d <= xmax
                fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
                ax.plot(d[sel], vr[sel], color=FIT_COLORS[0], lw=2,
                        label=f"circular rotation (flat, R0 {args.r0:g} kpc, "
                              f"Theta0 {args.theta0:g} km/s)")
                ax.axhline(v_src, color=FIT_COLORS[1], lw=2,
                           label=f"measured velocity ({v_src:.1f} km/s)")
                for x in d_kin:
                    ax.plot([x], [v_src], "o", ms=8, color=FIT_COLORS[1], mec=FIT_SURF, mew=2, zorder=5)
                    ax.annotate(f"kinematic distance {x:.1f} kpc", (x, v_src), xytext=(8, 8),
                                textcoords="offset points", fontsize=9)
                if args.parallax:
                    v_par = float(np.interp(args.parallax, d, vr))
                    print(f"At the parallax distance {args.parallax:g} kpc, circular rotation "
                          f"predicts {v_par:.1f} km/s -> non-circular motion "
                          f"{v_par - v_src:+.1f} km/s")
                    ax.axvline(args.parallax, color=FIT_INK2, lw=1.2, ls="--",
                               label=f"parallax distance ({args.parallax:g} kpc)")
                    ax.plot([args.parallax], [v_par], "o", ms=8, color=FIT_COLORS[0],
                            mec=FIT_SURF, mew=2, zorder=5)
                    ax.annotate("", xy=(args.parallax, v_src), xytext=(args.parallax, v_par),
                                arrowprops=dict(arrowstyle="<->", color=FIT_INK, lw=1.2))
                    ax.text(args.parallax + 0.08 * xmax / 6, (v_src + v_par) / 2,
                            f"{abs(v_par - v_src):.0f} km/s\nnon-circular motion",
                            va="center", fontsize=9)
                    ax.annotate(f"rotation predicts {v_par:.0f} km/s", (args.parallax, v_par),
                                xytext=(8, 8), textcoords="offset points", fontsize=9)
                ax.set_xlim(0, xmax)
                ax.set_xlabel("distance from the Sun along the line of sight (kpc)")
                ax.set_ylabel("LSR velocity (km/s)")
                ax.set_title(f"Kinematic distance: l = {np.degrees(l):.1f} deg, "
                             f"b = {np.degrees(b):.1f} deg", loc="left", fontsize=11)
                ax.legend(fontsize=8, frameon=False, loc="best")
                fig.savefig(out_base + "_kinematic.png", dpi=140)
                print(f"Saved plot : {out_base}_kinematic.png")

    if not args.no_show:
        plt.show()


# --------------------------------------------------------------------------
# Extra: WAV export for SDR#
# --------------------------------------------------------------------------
def cmd_wav(args):
    rec = load_recording(args.file, args.center, args.rate, args.start)
    describe(rec)
    if rec["datatype"] not in ("ci16_le", "ci16"):
        sys.exit("WAV export only supports 16-bit little-endian IQ (ci16_le).")
    fs = int(round(rec["fs"]))
    n = rec["n_samples"]
    if args.seconds:
        n = min(n, int(args.seconds * fs))
    nbytes = n * 4
    if nbytes + 36 > 0xFFFFFFFF:
        n = (0xFFFFFFFF - 36) // 4
        nbytes = n * 4
        print(f"Truncating to {n / fs:.1f} s to stay under the 4 GB WAV limit.")

    if args.out:
        out = args.out
    else:
        # SDR# reads the tuning frequency from file names in this pattern
        st = rec["start"] or datetime.now(timezone.utc)
        out = f"SDRSharp_{st:%Y%m%d_%H%M%S}Z_{int(round(rec['fc']))}Hz_IQ.wav"

    header = struct.pack("<4sI4s4sIHHIIHH4sI",
                         b"RIFF", 36 + nbytes, b"WAVE",
                         b"fmt ", 16, 1, 2, fs, fs * 4, 4, 16,
                         b"data", nbytes)
    raw = np.memmap(rec["data_path"], dtype="<i2", mode="r", offset=rec["header_bytes"])
    step = 16 * 1024 * 1024                         # values per chunk
    total_vals = 2 * n
    with open(out, "wb") as f:
        f.write(header)
        for a in range(0, total_vals, step):
            c = np.array(raw[a: min(a + step, total_vals)])
            if args.swap_iq:
                c = c.reshape(-1, 2)[:, ::-1].ravel()
            f.write(c.astype("<i2").tobytes())
            print(f"\r  writing {100 * min(a + step, total_vals) / total_vals:5.1f}%",
                  end="", flush=True)
    print(f"\nSaved {out}  ({n / fs:.1f} s, {fs} samples/s, centre {rec['fc'] / 1e6:.4f} MHz)")
    print("In SDR#: Source -> IQ File (*.wav), open this file. If the spectrum looks "
          "mirrored, re-run with --swap-iq.")


def cmd_info(args):
    rec = load_recording(args.file, args.center, args.rate, args.start)
    describe(rec)
    print("\nFull metadata:")
    print(json.dumps(rec["meta"], indent=2))


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--center", type=float, help="override center frequency (Hz)")
        sp.add_argument("--rate", type=float, help="override sample rate (samples/s)")
        sp.add_argument("--start", help="override start time, ISO format UTC "
                                        "(e.g. 2022-06-24T10:00:00Z)")

    sp = sub.add_parser("info", help="print what's in the metadata")
    sp.add_argument("file")
    common(sp)
    sp.set_defaults(func=cmd_info)

    def analysis(sp, vmin, vmax):
        """Options shared by 'spectrum' and 'stokes'."""
        common(sp)
        sp.add_argument("--nfft", type=int, default=16384,
                        help="FFT size; resolution = sample rate / nfft (default 16384 -> ~117 Hz)")
        sp.add_argument("--seconds", type=float, help="only process this many seconds")
        sp.add_argument("--skip", type=float, default=0.0,
                        help="skip this many seconds at the start")
        sp.add_argument("--rest", type=float, default=DEFAULT_REST_HZ,
                        help="rest frequency in Hz (default OH 1665.4018e6; HI is 1420.405751768e6)")
        sp.add_argument("--ra", default=DEFAULT_RA, help="target RA, J2000 (default W3(OH))")
        sp.add_argument("--dec", default=DEFAULT_DEC, help="target Dec, J2000 (default W3(OH))")
        sp.add_argument("--site", type=float, nargs=3, default=DEFAULT_SITE,
                        metavar=("LAT", "LON", "HEIGHT_M"),
                        help="observatory location (default Allen Telescope Array)")
        sp.add_argument("--no-lsr", action="store_true",
                        help="skip the LSR velocity correction")
        sp.add_argument("--vmin", type=float, default=vmin,
                        help=f"velocity plot window, km/s (default {vmin:g})")
        sp.add_argument("--vmax", type=float, default=vmax,
                        help=f"velocity plot window, km/s (default {vmax:g})")
        sp.add_argument("--baseline-khz", type=float, default=300.0,
                        help="running-median width for the baseline, kHz (default 300; "
                             "must be much wider than the line)")
        sp.add_argument("--edge", type=float, default=0.06,
                        help="fraction of each band edge to ignore (default 0.06)")
        sp.add_argument("--dc-mask", type=float, default=1500.0,
                        help="ignore +/- this many Hz around the DC spike (default 1500)")
        sp.add_argument("--out", help="output file name prefix")
        sp.add_argument("--no-show", action="store_true",
                        help="save files without opening a window")

    sp = sub.add_parser("spectrum", help="average, baseline-correct, plot")
    sp.add_argument("files", nargs="+", help="one .sigmf-meta (or -data) per polarization")
    analysis(sp, -100.0, 10.0)
    sp.add_argument("--dump", type=float, default=10.0,
                    help="seconds per row of the dynamic spectrum (default 10)")
    sp.add_argument("--row-norm", choices=["scale", "median", "none"], default="scale",
                    help="dynamic-spectrum row normalization: 'scale' (default) removes "
                         "broadband gain jumps; 'median' fits each row's own baseline "
                         "(slower); 'none' = old behaviour")
    sp.set_defaults(func=cmd_spectrum)

    sp = sub.add_parser("stokes", help="cross-correlate X and Y: Stokes I, Q, U, V per feature")
    sp.add_argument("xfile", help="X polarization .sigmf-meta (or -data)")
    sp.add_argument("yfile", help="Y polarization .sigmf-meta (or -data)")
    analysis(sp, -50.0, -40.0)
    sp.add_argument("--xy-phase", type=float, default=None, metavar="DEG",
                    help="instrumental phase that the receiver adds to XY*, in degrees; it is "
                         "subtracted. Unknown for this dataset; without it U and V are mixed")
    sp.add_argument("--features", type=float, nargs="+", metavar="V",
                    help="feature velocities to measure (km/s). Default: find peaks automatically")
    sp.add_argument("--halfwidth", type=float, default=0.15,
                    help="half-width of each feature's velocity window, km/s (default 0.15)")
    sp.add_argument("--min-snr", type=float, default=6.0,
                    help="automatic peak finding threshold in sigma (default 6)")
    sp.set_defaults(func=cmd_stokes)

    sp = sub.add_parser("fit", help="Gaussian components + kinematic distance from a spectrum CSV")
    sp.add_argument("csv", help="CSV written by the 'spectrum' command")
    sp.add_argument("--column", help="profile column to fit (default: mean_line, or the only *_line)")
    sp.add_argument("--vmin", type=float, default=-48.0, help="fit window start, km/s (default -48)")
    sp.add_argument("--vmax", type=float, default=-42.5, help="fit window end, km/s (default -42.5)")
    sp.add_argument("--guess", type=float, nargs="+", default=[-46.2, -45.2, -44.9, -44.1],
                    metavar="V", help="starting component velocities, km/s "
                                      "(default -46.2 -45.2 -44.9 -44.1)")
    sp.add_argument("--max-extra", type=int, default=1,
                    help="extra components to try, added at the largest residual and kept "
                         "only if the BIC improves (default 1)")
    sp.add_argument("--center-range", type=float, default=0.4,
                    help="how far a component may move from its guess, km/s (default 0.4)")
    sp.add_argument("--min-fwhm", type=float, default=0.07, help="minimum FWHM, km/s (default 0.07)")
    sp.add_argument("--max-fwhm", type=float, default=2.5, help="maximum FWHM, km/s (default 2.5)")
    sp.add_argument("--noise", type=float, nargs="+", default=[-60, -50, -40, -30],
                    metavar="V", help="line-free velocity ranges for the noise, as pairs "
                                      "(default -60 -50 -40 -30)")
    sp.add_argument("--integration", type=float,
                    help="integration time in s, to compare the noise with the radiometer "
                         "equation (506.2 for the full W3(OH) recording)")
    sp.add_argument("--mass-amu", type=float, default=17.0,
                    help="molecular mass for the thermal-width comparison (default 17, OH)")
    sp.add_argument("--tkin", type=float, nargs="+", default=[50, 100, 150],
                    help="gas temperatures for the thermal-width comparison, K")
    sp.add_argument("--ra", default=DEFAULT_RA, help="target RA, J2000 (default W3(OH))")
    sp.add_argument("--dec", default=DEFAULT_DEC, help="target Dec, J2000 (default W3(OH))")
    sp.add_argument("--r0", type=float, default=8.15,
                    help="Sun-Galactic centre distance, kpc (default 8.15, Reid et al. 2019)")
    sp.add_argument("--theta0", type=float, default=236.0,
                    help="circular rotation speed, km/s (default 236, Reid et al. 2019)")
    sp.add_argument("--parallax", type=float, default=1.95,
                    help="independent (parallax) distance to compare, kpc; 0 to omit "
                         "(default 1.95, Xu et al. 2006)")
    sp.add_argument("--kin-velocity", type=float,
                    help="velocity for the kinematic distance (default: strongest component)")
    sp.add_argument("--no-kinematic", action="store_true", help="skip the kinematic-distance plot")
    sp.add_argument("--out", help="output file name prefix (default <csv name>_fit)")
    sp.add_argument("--no-show", action="store_true", help="save files without opening a window")
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("wav", help="convert to a WAV file SDR# can play")
    sp.add_argument("file")
    common(sp)
    sp.add_argument("--seconds", type=float, help="only export this many seconds")
    sp.add_argument("--swap-iq", action="store_true", help="swap I and Q (mirrors the spectrum)")
    sp.add_argument("--out", help="output .wav name")
    sp.set_defaults(func=cmd_wav)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()