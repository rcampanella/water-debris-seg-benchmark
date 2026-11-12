import sys, json, numpy as np
from pathlib import Path
Q, P = Path(sys.argv[1]), Path(sys.argv[2])
def rd(f):
    b = open(f, "rb").read(); ncol = len(b[:b.index(b"\n")].split())
    return np.array(b.split(), dtype=np.float64).reshape(-1, ncol)
out = {}
for s in ["00", "04"]:
    a = rd(Q/f"p{s}/navigation/ahrs.txt"); l = rd(Q/f"p{s}/lidar_front/imu.txt")
    cam = rd(P/("stereo/timestamp.txt" if s == "00" else f"seq{s}/stereo/timestamp.txt"))[:, 0]
    r = {}
    for name, t in [("ahrs", a[:, 0]), ("lidar_imu", l[:, 0]), ("camera", cam)]:
        d = np.diff(t)
        r[name] = dict(n=len(t), median_dt_ms=1e3*np.median(d), p99_dt_ms=1e3*np.percentile(d, 99), max_dt_ms=1e3*d.max(),
                       backwards=int((d < 0).sum()), duplicates=int((d == 0).sum()), gaps_over_50ms=int((d > 0.05).sum()),
                       start_minus_camera_s=float(t[0]-cam[0]), end_minus_camera_s=float(t[-1]-cam[-1]))
    wa = np.linalg.norm(a[:, 4:7], axis=1); wl = np.linalg.norm(l[:, 1:4], axis=1)
    ta, tl = a[:, 0], l[:, 0]; oka = np.r_[True, np.diff(ta) > 0]; okl = np.r_[True, np.diff(tl) > 0]
    t0, t1 = max(ta[0], tl[0]) + 5, min(ta[-1], tl[-1]) - 5
    dt = 0.002; grid = np.arange(t0, t1, dt)
    A = np.interp(grid, ta[oka], wa[oka]); L = np.interp(grid, tl[okl], wl[okl])
    win = int(120 / dt); maxlag = int(2 / dt); rows = []
    for i in range(0, len(grid) - win, win):
        x = A[i:i+win] - A[i:i+win].mean(); y = L[i:i+win] - L[i:i+win].mean()
        n = 2 * win; c = np.fft.irfft(np.fft.rfft(y, n) * np.conj(np.fft.rfft(x, n)), n)
        c = np.r_[c[-maxlag:], c[:maxlag+1]] / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12)
        k = int(np.argmax(c)) - maxlag   # y(t) ~ x(t - k): LiDAR-IMU signal lags AHRS by k samples
        rows.append(dict(t_min=round((grid[i]-grid[0])/60, 1), lag_ms=round(-k*dt*1e3, 1), corr=round(float(c[k+maxlag]), 3)))
    r["windows_120s"] = rows; out[s] = r
    print(f"== pohang{s}")
    for name in ["ahrs", "lidar_imu", "camera"]:
        print(f"  {name:9s}", {k: round(v, 3) if isinstance(v, float) else v for k, v in r[name].items()})
    print("  lag (ms; positive = AHRS timestamps later than LiDAR-IMU for the same motion), corr:")
    print("  ", [(w["t_min"], w["lag_ms"], w["corr"]) for w in rows])
json.dump(out, open(Q/"q2_ahrs_timing.json", "w"), indent=1)
