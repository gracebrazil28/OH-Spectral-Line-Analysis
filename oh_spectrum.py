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
    used_s = total_ffts * nfft / fs
    print(f"  integrated {used_s:.1f} s, {total_ffts:,} FFTs, "
          f"resolution {fs / nfft:.1f} Hz/bin")
    return freqs_off, mean_spec, dyn, dyn_t, used_s


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
    v_topo = radio_velocity_kms(f_rf, args.rest)
    mid_time = None
    if recs[0]["start"] is not None:
        mid_time = recs[0]["start"] + timedelta(seconds=args.skip + results[0][4] / 2)
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
        v = v_topo + corr
        vlabel = "LSRK velocity (km/s, radio)"
    else:
        v = v_topo
        vlabel = "Topocentric velocity (km/s, radio)"

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

    # ---------- plots ----------
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
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
    r0, spec0, dyn0, dyn_t0, _ = results[0]
    dyn_norm = dyn0 / baselines[0][None, :] - 1.0
    cols = np.flatnonzero(sel) if sel.any() else np.flatnonzero(~bad)
    cols = cols[np.argsort(v[cols])]
    img = dyn_norm[:, cols]
    vmin_c, vmax_c = np.nanpercentile(img, [1, 99.5])
    ax.imshow(img, aspect="auto", origin="lower", cmap="viridis",
              vmin=vmin_c, vmax=vmax_c,
              extent=[v[cols][0], v[cols][-1], dyn_t0[0] - args.dump / 2,
                      dyn_t0[-1] + args.dump / 2])
    ax.set_xlabel(vlabel)
    ax.set_ylabel("Time from start (s)")
    ax.set_title(f"3. Dynamic spectrum ({r0['name']}, {args.dump:g} s per row)")

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

    sp = sub.add_parser("spectrum", help="average, baseline-correct, plot")
    sp.add_argument("files", nargs="+", help="one .sigmf-meta (or -data) per polarization")
    common(sp)
    sp.add_argument("--nfft", type=int, default=16384,
                    help="FFT size; resolution = sample rate / nfft (default 16384 -> ~117 Hz)")
    sp.add_argument("--seconds", type=float, help="only process this many seconds")
    sp.add_argument("--skip", type=float, default=0.0, help="skip this many seconds at the start")
    sp.add_argument("--dump", type=float, default=10.0,
                    help="seconds per row of the dynamic spectrum (default 10)")
    sp.add_argument("--rest", type=float, default=DEFAULT_REST_HZ,
                    help="rest frequency in Hz (default OH 1665.4018e6; HI is 1420.405751768e6)")
    sp.add_argument("--ra", default=DEFAULT_RA, help="target RA, J2000 (default W3(OH))")
    sp.add_argument("--dec", default=DEFAULT_DEC, help="target Dec, J2000 (default W3(OH))")
    sp.add_argument("--site", type=float, nargs=3, default=DEFAULT_SITE,
                    metavar=("LAT", "LON", "HEIGHT_M"),
                    help="observatory location (default Allen Telescope Array)")
    sp.add_argument("--no-lsr", action="store_true", help="skip the LSR velocity correction")
    sp.add_argument("--vmin", type=float, default=-100.0, help="velocity plot window (km/s)")
    sp.add_argument("--vmax", type=float, default=10.0, help="velocity plot window (km/s)")
    sp.add_argument("--baseline-khz", type=float, default=300.0,
                    help="running-median width for the baseline, kHz (default 300; "
                         "must be much wider than the line)")
    sp.add_argument("--edge", type=float, default=0.06,
                    help="fraction of each band edge to ignore (default 0.06)")
    sp.add_argument("--dc-mask", type=float, default=1500.0,
                    help="ignore +/- this many Hz around the DC spike (default 1500)")
    sp.add_argument("--out", help="output file name prefix")
    sp.add_argument("--no-show", action="store_true", help="save files without opening a window")
    sp.set_defaults(func=cmd_spectrum)

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