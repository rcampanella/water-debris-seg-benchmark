#!/usr/bin/env python
"""Depth ground truth for the water-surface depth benchmark (eval_outputs/water_debris_depth/README.md § Ground truth).

Versions
  pohang  uhrig : Uhrig et al. 2017 steps 1-6 (IMU de-skew, 11-scan accumulation with the released SLAM trajectory,
                  projection, SGM stereo cleaning with the D14 rule, water filter D11, range (min, 20] m)
  pohang  raw   : one sweep, no de-skew / accumulation / stereo; per-pixel conflicts -> point closest to image time
  multiaqua raw : one VLP-16 sweep; pixels hit by more than one point dropped
Output: uint16 PNG at native image resolution, value = round(depth_m * 256), 0 = invalid (KITTI / Ultralytics depth),
plus one JSON per frame with the points removed by each step per depth band.

Sub-commands
  pohang     build Pohang GT for a list of stereo frames (both versions in one pass)
  multiaqua  build MULTIAQUA raw GT for a list of frames
  summarize  coverage summary, alignment check, per-frame JSONL and overlay renders for the G1 evidence folder

Every choice that the README leaves open is a command-line parameter (see --help) and is written into each JSON.
"""
import argparse, io, json, math, os, sys, time, zipfile
import numpy as np
import cv2

BANDS = (0, 5, 10, 15, 20)             # 0-5 / 5-10 / 10-15 / 15-20 m, then > 20 m
BAND_NAMES = ["0-5", "5-10", "10-15", "15-20", ">20"]
DISP_BINS = (0, 60, 72, 80, 90, 100, 120, 145, 180, 240, 360, 800)   # px; ~fB/d m with fB ~ 1436
DEPTH_SCALE = 256.0

# reason codes (first step that removes a point, README step order)
KEPT, CONFLICT_PROJ, SGM_INVALID, TAU, WATER, MIN_RANGE, GT_MAX, CONFLICT = 0, 1, 2, 3, 4, 5, 6, 7
REASON = {KEPT: "kept", CONFLICT_PROJ: "pixel_conflict_projected", SGM_INVALID: "sgm_invalid", TAU: "stereo_tau",
          WATER: "water", MIN_RANGE: "range_min", GT_MAX: "range_gt_max", CONFLICT: "pixel_conflict_filtered"}


# ----------------------------------------------------------------------------------------------------------- geometry
def quat2R(q):
    """Quaternion [x, y, z, w] -> rotation matrix (Pohang extrinsics / baseline order)."""
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def project(Pc, K, D, wh):
    """Camera-frame points -> (u, v) float pixel coords (OpenCV convention: pixel centres at integers) and a mask of
    points in front of the camera and inside the image after rounding. D=None -> pinhole (undistorted frame)."""
    w, h = wh
    Z = Pc[:, 2]
    ok = Z > 0.1
    xn = np.where(ok, Pc[:, 0] / np.where(ok, Z, 1), 0)
    yn = np.where(ok, Pc[:, 1] / np.where(ok, Z, 1), 0)
    # generous normalised-coordinate gate: avoids the distortion polynomial folding far off-image points back in
    ok &= (np.abs(xn) < 1.5 * max(K[0, 2], w - K[0, 2]) / K[0, 0]) & (np.abs(yn) < 1.5 * max(K[1, 2], h - K[1, 2]) / K[1, 1])
    u = np.full(len(Pc), -1.0); v = np.full(len(Pc), -1.0)
    if ok.any():
        if D is None or not np.any(D):
            u[ok] = K[0, 0] * xn[ok] + K[0, 2]; v[ok] = K[1, 1] * yn[ok] + K[1, 2]
        else:
            uv, _ = cv2.projectPoints(Pc[ok].astype(np.float64), np.zeros(3), np.zeros(3), K, np.asarray(D, float))
            u[ok], v[ok] = uv[:, 0, 0], uv[:, 0, 1]
    ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    ok &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    return u, v, ok


def tau_outlier(d_lidar, d_sgm, px=3.0, rel=0.05):
    """D14 / KITTI stereo outlier rule: |d_l - d_s| >= px AND >= rel * d_l (relative to the LiDAR disparity)."""
    e = np.abs(d_lidar - d_sgm)
    return (e >= px) & (e >= rel * d_lidar)


def band_counts(Z, mask=None):
    Z = Z if mask is None else Z[mask]
    idx = np.searchsorted(np.array(BANDS[1:]), Z, side="left")   # (0,5] -> 0 ... (15,20] -> 3, >20 -> 4
    return np.bincount(np.clip(idx, 0, 4), minlength=5).tolist()


def encode_depth_png(depth_m):
    """float depth (0 / nan = invalid) -> uint16 array, value = round(depth * 256)."""
    d = np.nan_to_num(np.asarray(depth_m, float), nan=0.0)
    v = np.rint(d * DEPTH_SCALE)
    if v.max(initial=0) > 65535: raise ValueError("depth exceeds uint16 range")
    return v.astype(np.uint16)


def decode_depth_png(arr):
    return arr.astype(np.float32) / DEPTH_SCALE


def write_depth_png(path, depth_img):
    ok = cv2.imwrite(path, encode_depth_png(depth_img))
    if not ok: raise IOError(path)


# ------------------------------------------------------------------------------------------------- pixel conflicts
def conflict_drop_all(lin, cand):
    """MULTIAQUA rule: among candidate points, pixels hit by > 1 point lose all their points. Returns removed mask."""
    rem = np.zeros(len(lin), bool)
    idx = np.flatnonzero(cand)
    if len(idx) == 0: return rem
    _, inv, cnt = np.unique(lin[idx], return_inverse=True, return_counts=True)
    rem[idx[cnt[inv] > 1]] = True
    return rem


def conflict_closest_time(lin, absdt, cand):
    """Eigen et al. 2014 rule: among candidate points in one pixel keep the one recorded closest to the image time.
    Exact time ties -> all tied points dropped if they are the minimum (conservative). Returns removed mask."""
    rem = np.zeros(len(lin), bool)
    idx = np.flatnonzero(cand)
    if len(idx) == 0: return rem
    o = idx[np.lexsort((absdt[idx], lin[idx]))]
    l = lin[o]; first = np.r_[True, l[1:] != l[:-1]]
    gstart = np.flatnonzero(first); gid = np.cumsum(first) - 1
    keep = first.copy()
    # tie at the minimum -> drop the whole pixel
    mint = absdt[o][gstart][gid]
    nmin = np.bincount(gid, weights=(absdt[o] == mint).astype(float))
    keep &= nmin[gid] == 1
    rem[o[~keep]] = True
    return rem


def conflict_uhrig(lin, Z, absdt, cand, fB, px=3.0, rel=0.05):
    """Accumulated-scan pixels (README leaves open; conservative choice): if the surviving points of a pixel disagree
    by the D14 rule (max - min disparity >= px and >= rel * min disparity) the pixel is dropped, otherwise the point
    closest to the image time is kept. Returns removed mask."""
    rem = np.zeros(len(lin), bool)
    idx = np.flatnonzero(cand)
    if len(idx) == 0: return rem
    o = idx[np.lexsort((absdt[idx], lin[idx]))]
    l = lin[o]; first = np.r_[True, l[1:] != l[:-1]]
    gs = np.flatnonzero(first); gid = np.cumsum(first) - 1
    d = fB / Z[o]
    dmax = np.maximum.reduceat(d, gs); dmin = np.minimum.reduceat(d, gs)
    dis = ((dmax - dmin) >= px) & ((dmax - dmin) >= rel * dmin)
    mint = absdt[o][gs][gid]
    nmin = np.bincount(gid, weights=(absdt[o] == mint).astype(float))
    keep = first & ~dis[gid] & (nmin[gid] == 1)
    rem[o[~keep]] = True
    return rem


# -------------------------------------------------------------------------------------------------------- water mask
class WaterMask:
    """D11: union of 'Water' instance masks of the LaRS-trained YOLO26m-seg, grown by `grow` px (square kernel)."""
    def __init__(self, weights, imgsz=640, conf=0.25, grow=10, device=None):
        from ultralytics import YOLO
        self.m = YOLO(weights); self.imgsz, self.conf, self.grow, self.device = imgsz, conf, grow, device
        self.names = self.m.names
        self.water_ids = [k for k, n in self.names.items() if n.lower() == "water"]
        self.boat_ids = [k for k, n in self.names.items() if any(s in n.lower() for s in ("boat", "ship"))]
        assert self.water_ids, self.names

    def __call__(self, img_bgr):
        r = self.m.predict(img_bgr, imgsz=self.imgsz, conf=self.conf, retina_masks=True, verbose=False,
                           device=self.device)[0]
        water = np.zeros(img_bgr.shape[:2], bool); boats = 0
        if r.masks is not None:
            cls = r.boxes.cls.int().tolist(); md = r.masks.data.cpu().numpy()
            for c, mk in zip(cls, md):
                if c in self.water_ids: water |= mk > 0.5
            boats = sum(c in self.boat_ids for c in cls)
        grown = cv2.dilate(water.astype(np.uint8), np.ones((2 * self.grow + 1,) * 2, np.uint8)) > 0 if self.grow else water
        return water, grown, boats


# ----------------------------------------------------------------------------------------------------- Pohang source
class Pohang:
    def __init__(self, raw_dir, a):
        self.raw, self.a = raw_dir, a
        with zipfile.ZipFile(f"{raw_dir}/calibration.zip") as z:
            nm = {os.path.basename(n): n for n in z.namelist()}
            E = json.loads(z.read(nm["extrinsics.json"])); I = json.loads(z.read(nm["intrinsics.json"]))
        with zipfile.ZipFile(f"{raw_dir}/navigation.zip") as z:
            self.base = np.loadtxt(io.BytesIO(z.read("navigation/baseline.txt")))
        # sensor -> body (AHRS, FRD) poses; quaternion [x, y, z, w]
        self.R_bl, self.t_bl = quat2R(E["lidar_front"]["quaternion"]), np.array(E["lidar_front"]["translation"])
        self.R_bc, self.t_bc = quat2R(E["stereo_left"]["quaternion"]), np.array(E["stereo_left"]["translation"])
        R_br, t_br = quat2R(E["stereo_right"]["quaternion"]), np.array(E["stereo_right"]["translation"])
        def K_(c): return np.array([[c["focal_length"], 0, c["cc_x"]], [0, c["focal_length"], c["cc_y"]], [0, 0, 1.]]), \
            np.array(c["distortion_coefficients"], float)
        self.K, self.D = K_(I["stereo_left"]); Kr, Dr = K_(I["stereo_right"])
        self.wh = (I["stereo_left"]["image_width"], I["stereo_left"]["image_height"])
        # left-camera -> right-camera (x_r = R x_l + T) and rectification (rectified frame used for SGM only)
        R_lr = R_br.T @ self.R_bc; T_lr = R_br.T @ (self.t_bc - t_br)
        self.baseline_m = float(np.linalg.norm(T_lr))
        R1, R2, P1, P2, *_ = cv2.stereoRectify(self.K, self.D, Kr, Dr, self.wh, R_lr, T_lr,
                                               flags=cv2.CALIB_ZERO_DISPARITY, alpha=a.rect_alpha)
        self.R1, self.P1 = R1, P1
        self.fB = float(-P2[0, 3])                       # f_rect * B (px * m)
        self.map_l = cv2.initUndistortRectifyMap(self.K, self.D, R1, P1, self.wh, cv2.CV_32FC1)
        self.map_r = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, self.wh, cv2.CV_32FC1)
        self.map_und = cv2.initUndistortRectifyMap(self.K, self.D, None, self.K, self.wh, cv2.CV_32FC1)
        # GT frame: undistorted left camera (README step 3, default) or the released (distorted) image
        self.Dgt = None if a.gt_frame == "undistorted" else self.D
        # static lidar -> camera
        self.R_cl = self.R_bc.T @ self.R_bl; self.t_cl = self.R_bc.T @ (self.t_bl - self.t_bc)
        self.cal_info = {"baseline_m": self.baseline_m, "f_rect_px": float(P1[0, 0]), "fB_px_m": self.fB,
                         "rect_alpha": a.rect_alpha, "gt_frame": a.gt_frame,
                         "K_left": self.K.tolist(), "D_left": self.D.tolist()}
        self._zs = self._zl = None
        self._imu = None

    # lazily opened per process
    @property
    def zs(self):
        if self._zs is None: self._zs = zipfile.ZipFile(f"{self.raw}/stereo.zip")
        return self._zs

    @property
    def zl(self):
        if self._zl is None:
            self._zl = zipfile.ZipFile(f"{self.raw}/lidar/lidar_front.zip")
            bins = sorted(n for n in self._zl.namelist() if n.endswith(".bin"))
            self.scan_names = bins; self.scan_ts = np.array([int(n.rsplit("/", 1)[1][:-4]) for n in bins]) / 1e9
        return self._zl

    def cam_times(self):
        return np.array([float(l.split()[0]) for l in self.zs.read("stereo/timestamp.txt").decode().splitlines() if l.strip()])

    def imu(self):
        if self._imu is None:
            arr = np.loadtxt(io.StringIO(self.zl.read("lidar_front/imu.txt").decode()))
            t = arr[:, 0]; g_b = arr[:, 1:4] @ self.R_bl.T       # gyro (rad/s), LiDAR-IMU frame -> body
            from scipy.spatial.transform import Rotation as Rot
            tb = self.base[:, 0]; Rw = Rot.from_quat(self.base[:, 1:5])
            rv = (Rw[:-1].inv() * Rw[1:]).as_rotvec()           # SLAM body-frame increments
            dt = np.diff(t); cg = np.vstack([np.zeros(3), np.cumsum(0.5 * (g_b[1:] + g_b[:-1]) * dt[:, None], 0)])
            self._imu = dict(t=t, g=g_b, cg=cg, tb=tb, rv=rv, crv=np.vstack([np.zeros(3), np.cumsum(rv, 0)]), Rw=Rw)
        return self._imu

    def gyro_bias(self, tc):
        """Gyro bias (body frame, rad/s): mean(gyro) - mean(SLAM angular rate) over +- W/2 around tc (README leaves
        bias handling open; LIO-SAM / FAST-LIO estimate it online)."""
        m = self.imu(); W = self.a.gyro_bias_window_s
        i0 = np.searchsorted(m["tb"], tc - W / 2); i1 = np.searchsorted(m["tb"], tc + W / 2) - 1
        i0 = max(i0, 0); i1 = min(i1, len(m["tb"]) - 1)
        if i1 - i0 < 5: return np.zeros(3)
        T = m["tb"][i1] - m["tb"][i0]
        G = np.array([np.interp(m["tb"][i1], m["t"], m["cg"][:, j]) - np.interp(m["tb"][i0], m["t"], m["cg"][:, j]) for j in range(3)])
        S = m["crv"][i1] - m["crv"][i0]
        return (G - S) / T

    def imu_rel_rot(self, tc, tq, bias):
        """Rotations R_b(tc)^T R_b(tq) from bias-corrected gyro integration (trapezoid), as scipy Rotation."""
        from scipy.spatial.transform import Rotation as Rot, Slerp
        m = self.imu(); t = m["t"]
        lo, hi = min(tq.min(), tc) - 0.05, max(tq.max(), tc) + 0.05
        i0 = max(np.searchsorted(t, lo) - 1, 0); i1 = min(np.searchsorted(t, hi) + 1, len(t))
        tt = t[i0:i1]; w = m["g"][i0:i1] - bias
        if tt[0] > lo + 0.04 or tt[-1] < hi - 0.04: raise ValueError("IMU does not cover the scan window")
        inc = Rot.from_rotvec(0.5 * (w[1:] + w[:-1]) * np.diff(tt)[:, None])
        cum = [Rot.identity()]
        for r in inc: cum.append(cum[-1] * r)
        sl = Slerp(tt, Rot.concatenate(cum))
        gap = float(np.diff(tt).max())
        return sl([tc])[0].inv() * sl(np.clip(tq, tt[0], tt[-1])), gap

    def slam_pose(self, tq):
        from scipy.spatial.transform import Slerp
        m = self.imu(); tb = m["tb"]
        if np.any(tq < tb[0]) or np.any(tq > tb[-1]): raise ValueError("SLAM trajectory does not cover the window")
        pos = np.stack([np.interp(tq, tb, self.base[:, 5 + j]) for j in range(3)], 1)
        return Slerp(tb, m["Rw"])(tq), pos

    def read_scan(self, k):
        r = np.frombuffer(self.zl.read(self.scan_names[k]), np.uint8).reshape(-1, 28)
        f = r[:, :16].copy().view(np.float32); off = r[:, 16:20].copy().view(np.uint32).ravel()
        keep = np.abs(f[:, :3]).sum(1) > 0
        return f[keep, :3].astype(np.float64), self.scan_ts[k] + off[keep] / 1e9

    def fov_time_offset(self, n=20):
        """Median per-point time offset (s from the file timestamp) of points that land in the image (static)."""
        _ = self.zl; offs = []
        for k in np.linspace(100, len(self.scan_names) - 100, n).astype(int):
            P, tp = self.read_scan(k)
            _, _, ok = project(P @ self.R_cl.T + self.t_cl, self.K, self.D, self.wh)
            if ok.sum() > 100: offs.append(np.median(tp[ok] - self.scan_ts[k]))
        return float(np.median(offs))


def sgm_disparity(left_rect, right_rect, a):
    """OpenCV SGBM (Hirschmueller SGM) on grey rectified images. Both images are padded on the left by numDisparities
    columns so that the left border also gets disparities. Invalid -> nan."""
    nd = a.sgm_num_disp; bs = a.sgm_block
    L = cv2.copyMakeBorder(left_rect, 0, 0, nd, 0, cv2.BORDER_CONSTANT, 0)
    R = cv2.copyMakeBorder(right_rect, 0, 0, nd, 0, cv2.BORDER_CONSTANT, 0)
    mode = {"sgbm": cv2.STEREO_SGBM_MODE_SGBM, "hh": cv2.STEREO_SGBM_MODE_HH, "3way": cv2.STEREO_SGBM_MODE_SGBM_3WAY}[a.sgm_mode]
    sg = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=bs, P1=8 * bs * bs, P2=32 * bs * bs,
                               disp12MaxDiff=a.sgm_lr_max_diff, uniquenessRatio=a.sgm_uniqueness,
                               speckleWindowSize=a.sgm_speckle_window, speckleRange=a.sgm_speckle_range, mode=mode)
    d = sg.compute(L, R)[:, nd:].astype(np.float32) / 16.0
    d[d <= 0] = np.nan
    return d


# per-process globals for multiprocessing workers
_G = {}


def _init_pohang(raw_dir, a):
    cv2.setNumThreads(a.cv_threads)
    _G["src"] = Pohang(raw_dir, a); _G["a"] = a
    _ = _G["src"].zl; _G["src"].imu(); _G["fov_off"] = a._fov_off


def pohang_frame(item):
    """CPU part for one stereo frame: everything except the water mask (needs the GPU model, done by the parent)."""
    fi, tc = item
    S, a = _G["src"], _G["a"]; t0 = time.time()
    out = {"frame": f"{fi:06d}", "t_cam": tc, "versions": {}, "errors": []}
    imgL = cv2.imdecode(np.frombuffer(S.zs.read(f"stereo/left_images/{fi:06d}.png"), np.uint8), cv2.IMREAD_COLOR)
    imgR = cv2.imdecode(np.frombuffer(S.zs.read(f"stereo/right_images/{fi:06d}.png"), np.uint8), cv2.IMREAD_COLOR)
    gt_img = cv2.remap(imgL, *S.map_und, cv2.INTER_LINEAR) if a.gt_frame == "undistorted" else imgL
    out["image"] = gt_img
    # scan association: in-FOV time (file ts + median in-FOV offset) closest to the image time
    fov_t = S.scan_ts + _G["fov_off"]
    k = int(np.argmin(np.abs(fov_t - tc)))
    out["centre_scan"] = {"index": k, "file_ts": float(S.scan_ts[k]), "file_ts_minus_cam_ms": (S.scan_ts[k] - tc) * 1e3,
                          "fov_time_minus_cam_ms": (fov_t[k] - tc) * 1e3}
    W, H = S.wh
    # ---------- raw: one sweep, static extrinsics, no de-skew
    P, tp = S.read_scan(k)
    rng = np.linalg.norm(P, axis=1)
    Pc = P @ S.R_cl.T + S.t_cl
    u, v, ok = project(Pc, S.K, S.Dgt, S.wh)
    raw = dict(u=u[ok], v=v[ok], Z=Pc[ok, 2], absdt=np.abs(tp[ok] - tc), rng=rng[ok], rel=np.zeros(ok.sum(), np.int8))
    raw["lin"] = np.rint(raw["v"]).astype(np.int64) * W + np.rint(raw["u"]).astype(np.int64)
    raw["reason"] = np.zeros(ok.sum(), np.int8)
    if a.raw_conflict_stage == "projected":
        raw["reason"][conflict_closest_time(raw["lin"], raw["absdt"], np.ones(ok.sum(), bool))] = CONFLICT_PROJ
    raw["fov_point_time_minus_cam_ms"] = float(np.median(tp[ok] - tc) * 1e3) if ok.any() else None
    out["versions"]["raw"] = raw
    if "uhrig" not in a.versions: return out
    # ---------- uhrig: de-skew + accumulation
    ks = [k + j for j in range(-a.n_scans // 2 + 1, a.n_scans // 2 + 1)]
    if ks[0] < 0 or ks[-1] >= len(S.scan_names):
        out["errors"].append("uhrig: scan window outside sequence"); return out
    Ps, ts, rels = [], [], []
    for kk in ks:
        Pk, tk = (P, tp) if kk == k else S.read_scan(kk)
        # coarse gate with the static extrinsics (wide margin) to avoid transforming points far outside the image
        Pck = Pk @ S.R_cl.T + S.t_cl
        g = (Pck[:, 2] > 0.1) & (np.abs(Pck[:, 0]) < 1.2 * Pck[:, 2] + 2) & (np.abs(Pck[:, 1]) < 0.9 * Pck[:, 2] + 2)
        Ps.append(Pk[g]); ts.append(tk[g]); rels.append(np.full(g.sum(), kk - k, np.int8))
    P = np.concatenate(Ps); tp = np.concatenate(ts); rel = np.concatenate(rels); rng = np.linalg.norm(P, axis=1)
    Pb = P @ S.R_bl.T + S.t_bl                                    # body frame at t_p
    try:
        bias = S.gyro_bias(tc) if a.gyro_bias_window_s > 0 else np.zeros(3)
        Rwc, pc = S.slam_pose(np.array([tc]))
        Rwp, pp = S.slam_pose(tp)
        if a.rotation_source == "imu":
            dR, gap = S.imu_rel_rot(tc, tp, bias)
        else:
            dR, gap = Rwc[0].inv() * Rwp, None
        trans = (pp - pc) @ Rwc[0].as_matrix() if a.deskew_translation == "slam" else 0.0   # R_wc^T (p(tp) - p(tc))
        Pbc = dR.apply(Pb) + trans
    except ValueError as e:
        out["errors"].append(f"uhrig: {e}"); return out
    # de-skew effect on the centre scan (pixel shift, raw vs de-skewed) for the report
    Pcam = (Pbc - S.t_bc) @ S.R_bc
    u, v, ok = project(Pcam, S.K, S.Dgt, S.wh)
    u0, v0, ok0 = project((Pb - S.t_bc) @ S.R_bc, S.K, S.Dgt, S.wh)
    c = ok & ok0 & (rel == 0) & (Pcam[:, 2] <= a.max_depth)
    shift = np.hypot(u - u0, v - v0)[c]
    uh = dict(u=u[ok], v=v[ok], Z=Pcam[ok, 2], absdt=np.abs(tp[ok] - tc), rng=rng[ok], rel=rel[ok],
              reason=np.zeros(ok.sum(), np.int8))
    uh["lin"] = np.rint(uh["v"]).astype(np.int64) * W + np.rint(uh["u"]).astype(np.int64)
    uh["motion"] = {"gyro_bias_rad_s": bias.tolist(), "imu_max_gap_s": gap, "rotation_source": a.rotation_source,
                    "deskew_translation": a.deskew_translation,
                    "centre_scan_shift_px_le_max_depth": {"n": int(len(shift)),
                        "median": float(np.median(shift)) if len(shift) else None,
                        "p95": float(np.percentile(shift, 95)) if len(shift) else None}}
    # ---------- stereo cleaning (D14)
    lr = cv2.remap(cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), *S.map_l, cv2.INTER_LINEAR)
    rr = cv2.remap(cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY), *S.map_r, cv2.INTER_LINEAR)
    disp = sgm_disparity(lr, rr, a)
    Pr = Pcam[ok] @ S.R1.T
    ur = S.P1[0, 0] * Pr[:, 0] / Pr[:, 2] + S.P1[0, 2]; vr = S.P1[1, 1] * Pr[:, 1] / Pr[:, 2] + S.P1[1, 2]
    ui, vi = np.rint(ur).astype(int), np.rint(vr).astype(int)
    inr = (Pr[:, 2] > 0) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ds = np.full(len(Pr), np.nan); ds[inr] = disp[vi[inr], ui[inr]]
    ds = ds + a.stereo_disp_offset_px   # G1 diagnostic: constant stereo calibration offset (default 0)
    dl = S.fB / Pr[:, 2]
    valid = np.isfinite(ds)
    out_tau = np.zeros(len(Pr), bool); out_tau[valid] = tau_outlier(dl[valid], ds[valid], a.tau_px, a.tau_rel)
    if a.sgm_invalid == "drop": uh["reason"][~valid] = SGM_INVALID
    uh["reason"][valid & out_tau] = TAU
    uh["d_lidar"], uh["d_sgm"] = dl, ds
    out["sgm_valid_frac_image"] = float(np.isfinite(disp).mean())
    out["versions"]["uhrig"] = uh
    out["cpu_s"] = time.time() - t0
    return out


# --------------------------------------------------------------------------------------------------- common finishing
def finish_version(V, water_grown, a, fB=None, version="raw", hull=None):
    """Apply water (step 5), range (step 6) and the filtered-stage pixel conflict; returns stats and depth image."""
    vi = np.rint(V["v"]).astype(int); ui = np.rint(V["u"]).astype(int)
    r = V["reason"]
    on_water = water_grown[vi, ui]
    if a.water_filter == "remove":                # D11 adaptation (later ablation); 'off' = unadapted GT, tag only
        m = (r == KEPT) & on_water; r[m] = WATER
    if hull is not None:
        m = (r == KEPT) & hull[vi, ui]; r[m] = MIN_RANGE         # own-hull pixels: counted with the range step
    m = (r == KEPT) & (V["rng"] <= a.min_range); r[m] = MIN_RANGE
    m = (r == KEPT) & (V["Z"] > a.max_depth); r[m] = GT_MAX
    cand = r == KEPT
    alt = None
    if version == "uhrig":
        r[conflict_uhrig(V["lin"], V["Z"], V["absdt"], cand, fB, a.tau_px, a.tau_rel)] = CONFLICT
    elif a.raw_conflict_stage == "filtered":
        rule = conflict_drop_all if a.raw_conflict == "drop" else (lambda l, c: conflict_closest_time(l, V["absdt"], c))
        r[rule(V["lin"], cand)] = CONFLICT
    if version != "uhrig":
        # alternative stage, for the report only: how many final pixels the other stage would give
        if a.raw_conflict_stage == "projected":
            pf = (V["rng"] > a.min_range) & (V["Z"] <= a.max_depth)
            if a.water_filter == "remove": pf &= ~on_water
            if hull is not None: pf &= ~hull[vi, ui]
            cand2 = ((r == KEPT) | (r == CONFLICT_PROJ)) & pf
            rule = conflict_drop_all if a.raw_conflict == "drop" else (lambda l, c: conflict_closest_time(l, V["absdt"], c))
            alt = int((cand2 & ~rule(V["lin"], cand2)).sum())
    keep = r == KEPT
    depth = np.zeros((water_grown.shape[0], water_grown.shape[1]), np.float32)
    depth[vi[keep], ui[keep]] = V["Z"][keep]
    st = {"projected": band_counts(V["Z"])}
    for code, name in REASON.items():
        if code == KEPT: continue
        st[f"removed_{name}"] = band_counts(V["Z"], r == code)
    st["final_points"] = band_counts(V["Z"], keep)
    st["final_points_on_water_mask"] = band_counts(V["Z"], keep & on_water)   # reflections / floating objects, for later analysis
    n20 = int((keep & (V["Z"] <= a.max_depth)).sum())
    res = {"stats_per_band": st, "n_final_le_max_depth": n20, "pass_min_points": n20 >= a.min_points}
    if alt is not None: res["n_final_if_conflict_stage_filtered"] = alt
    if version == "uhrig":
        rel = V["rel"]; vs = np.isfinite(V["d_sgm"]); le = V["Z"] <= a.max_depth
        res["sgm_valid_rate_at_lidar"] = {b: (float(vs[bm].mean()) if bm.sum() else None)
                                          for b, bm in zip(BAND_NAMES, _band_masks(V["Z"]))}
        res["n_lidar_le_max_depth"] = int(le.sum())
        # accumulation: points by scan offset (projected) and stereo outlier rate by scan offset (alignment check)
        acc = {}
        for j in np.unique(rel):
            mj = rel == j; mv = mj & vs & le & (r != WATER)
            acc[int(j)] = {"projected": int(mj.sum()), "projected_le_max_depth": int((mj & le).sum()),
                           "sgm_valid_le_max_depth_nonwater": int(mv.sum()),
                           "tau_outliers": int((mv & tau_outlier(V["d_lidar"], np.nan_to_num(V["d_sgm"]), a.tau_px, a.tau_rel)).sum()),
                           "final_points": int((mj & keep).sum())}
        res["by_scan_offset"] = acc
        e = (V["d_lidar"] - V["d_sgm"])[vs & le & (r != WATER)]
        res["stereo_residual_px_le_max_depth_nonwater"] = {
            "n": int(len(e)), "median": float(np.median(e)) if len(e) else None,
            "mad": float(np.median(np.abs(e - np.median(e)))) if len(e) else None,
            "hist_edges": [-10, 10, 0.5], "hist": np.histogram(np.clip(e, -10, 10), bins=40, range=(-10, 10))[0].tolist()}
        # residual (d_lidar - d_sgm) by LiDAR disparity bin: constant -> stereo yaw / offset, proportional -> baseline
        # scale, ~ d^2 -> depth offset (LiDAR-camera translation along the optical axis or timing)
        dl = V["d_lidar"][vs & le & (r != WATER)]
        res["stereo_residual_by_disparity_bin"] = [
            {"d_lo": lo, "d_hi": hi, "n": int(m.sum()), "median": float(np.median(e[m])) if m.sum() else None,
             "sum": float(e[m].sum()), "outliers": int(tau_outlier(dl[m], dl[m] - e[m], a.tau_px, a.tau_rel).sum())}
            for lo, hi in zip(DISP_BINS[:-1], DISP_BINS[1:]) for m in [(dl >= lo) & (dl < hi)]]
        res["motion"] = V["motion"]
    return res, depth, r


def _band_masks(Z):
    return [(Z > lo) & (Z <= hi) for lo, hi in zip(BANDS[:-1], BANDS[1:])] + [Z > BANDS[-1]]


def save_cache(path, V, r, img):
    """Small per-frame cache for renders: image at <= 1024 px and points with reason codes."""
    s = 1024 / img.shape[1]
    small = cv2.resize(img, (1024, int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path + ".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
    np.savez_compressed(path + ".npz", u=(V["u"] * s).astype(np.float32), v=(V["v"] * s).astype(np.float32),
                        Z=V["Z"].astype(np.float32), reason=r.astype(np.int8), rel=V["rel"].astype(np.int8))


def frame_tags(img, water, boats, V, r, a):
    keep = r == KEPT
    Z = V["Z"][keep]; vv = V["v"][keep]
    H = img.shape[0]
    return {"mean_brightness": float(img.mean()), "water_frac": float(water.mean()), "boat_instances": int(boats),
            "n_final_le10": int((Z <= 10).sum()), "n_final_15_20": int(((Z > 15) & (Z <= 20)).sum()),
            "n_final_upper_third": int((vv < H / 3).sum())}


def run_pohang(a):
    import multiprocessing as mp
    raw_dir = a.raw_dir; seq = a.seq
    S = Pohang(raw_dir, a)
    tcam = S.cam_times() + a.cam_time_offset_s
    frames = parse_frames(a.frames, len(tcam))
    if a.shard:
        i, n = map(int, a.shard.split("/")); frames = frames[i::n]
    a._fov_off = S.fov_time_offset()
    print(f"[pohang] {seq}: {len(frames)} frames, fov time offset {a._fov_off*1e3:.1f} ms, baseline {S.baseline_m:.4f} m, "
          f"fB {S.fB:.1f}", flush=True)
    outd = os.path.join(a.out, seq); os.makedirs(outd, exist_ok=True)
    for v in a.versions: os.makedirs(f"{outd}/{v}/depth", exist_ok=True)
    os.makedirs(f"{outd}/stats", exist_ok=True); os.makedirs(f"{outd}/cache", exist_ok=True)
    params = run_params(a); params["calibration"] = S.cal_info; params["fov_time_offset_s"] = a._fov_off
    json.dump(params, open(f"{outd}/params.json", "w"), indent=1)
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.workers, initializer=_init_pohang, initargs=(raw_dir, a))
    wm = WaterMask(a.water_model, grow=a.water_grow, device=a.device)
    done = 0; t0 = time.time()
    todo = [f for f in frames if not os.path.exists(f"{outd}/stats/{f:06d}.json")]   # resume after an interrupted run
    if len(todo) < len(frames): print(f"[pohang] resuming: {len(frames) - len(todo)} frames already built", flush=True)
    done = len(frames) - len(todo)
    batch = 4 * a.workers                       # bounded backlog: workers outpace the GPU water mask (OOM otherwise)
    def _results():
        for i in range(0, len(todo), batch):
            yield from pool.imap_unordered(pohang_frame, [(f, float(tcam[f])) for f in todo[i:i + batch]], chunksize=1)
    for res in _results():
        fr = res["frame"]; img = res.pop("image")
        water, grown, boats = wm(img)
        rec = {k: v for k, v in res.items() if k != "versions"}
        rec["sequence"] = seq; rec["versions"] = {}
        for ver in a.versions:
            if ver not in res["versions"]: rec["versions"][ver] = None; continue
            V = res["versions"][ver]
            st, depth, r = finish_version(V, grown, a, fB=S.fB, version=ver)
            if ver == "raw": st["fov_point_time_minus_cam_ms"] = V["fov_point_time_minus_cam_ms"]
            st["tags"] = frame_tags(img, water, boats, V, r, a)
            rec["versions"][ver] = st
            write_depth_png(f"{outd}/{ver}/depth/{fr}.png", depth)
            if int(fr) % a.cache_every == 0: save_cache(f"{outd}/cache/{fr}_{ver}", V, r, img)
        rec["water_frac_raw_mask"] = float(water.mean()); rec["water_frac_grown_mask"] = float(grown.mean())
        json.dump(rec, open(f"{outd}/stats/{fr}.json", "w"))
        done += 1
        if done % 10 == 0 or done == len(frames):
            print(f"[pohang] {done}/{len(frames)} {time.time()-t0:.0f}s", flush=True)
    pool.close(); pool.join()


# -------------------------------------------------------------------------------------------------- MULTIAQUA source
class MultiAqua:
    def __init__(self, zip_path):
        self.zp = zip_path; self._z = None
        z = self.z
        def ym(name):
            fs = cv2.FileStorage(); fs.open(z.read(name).decode(), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
            return fs
        L = ym("calibration/lidar_zed.yaml"); Zc = ym("calibration/zed.yaml")
        self.R, self.T, self.C = L.getNode("R").mat(), L.getNode("T").mat().ravel(), L.getNode("C").mat().astype(float)
        self.K = Zc.getNode("cameraMatrix").mat(); self.D = Zc.getNode("distCoeffs").mat().ravel()
        sz = Zc.getNode("imageSize"); self.wh = (int(sz.at(0).real()), int(sz.at(1).real()))

    @property
    def z(self):
        if self._z is None: self._z = zipfile.ZipFile(self.zp)
        return self._z

    def frames(self):
        return sorted(n.rsplit("/", 1)[1][:-4] for n in self.z.namelist() if n.startswith("data/zed/") and n.endswith(".png"))


def hull_masks_from_labels(S, a, outd):
    """Own-hull pixels (README step 6) for MULTIAQUA: per sequence, pixels left unlabelled (value 0; the recording
    boat's bow is not annotated) in >= hull_label_freq of that sequence's labelled frames, grown by water_grow px.
    Sequences without labels (lj4) get no mask. Pohang: bow not visible in the left camera (checked on renders)."""
    import collections
    if a.hull_label_freq <= 0: return {}
    ann = [n for n in S.z.namelist() if n.startswith("annotations/") and n.endswith(".png")]
    acc = {}; cnt = collections.Counter()
    for n in ann:
        sq = n.split("/")[1].split("_")[0]
        lab = cv2.imdecode(np.frombuffer(S.z.read(n), np.uint8), cv2.IMREAD_UNCHANGED)
        acc[sq] = acc.get(sq, 0) + (lab == 0).astype(np.float32); cnt[sq] += 1
    out = {}
    for sq in acc:
        m = (acc[sq] / cnt[sq]) >= a.hull_label_freq
        if a.water_grow: m = cv2.dilate(m.astype(np.uint8), np.ones((2 * a.water_grow + 1,) * 2, np.uint8)) > 0
        out[sq] = m
        cv2.imwrite(f"{outd}/hull_mask_{sq}.png", m.astype(np.uint8) * 255)
    return out


def _init_mq(zip_path, a):
    cv2.setNumThreads(a.cv_threads); _G["src"] = MultiAqua(zip_path); _G["a"] = a
    _G["names"] = set(_G["src"].z.namelist())


def mq_frame(fr):
    S, a = _G["src"], _G["a"]
    img = cv2.imdecode(np.frombuffer(S.z.read(f"data/zed/{fr}.png"), np.uint8), cv2.IMREAD_COLOR)
    out = {"frame": fr, "image": img, "versions": {}, "errors": []}
    n = f"data/lidar/{fr}.npy"
    if n not in _G["names"]:
        out["errors"].append("no lidar file"); P = np.zeros((0, 4))
    else:
        P = np.load(io.BytesIO(S.z.read(n)))
    P = P[np.abs(P[:, :3]).sum(1) > 0] if len(P) else P
    n_nonzero = len(P)
    if a.mq_dedupe and len(P):
        # observed in the release: every return is stored twice (rows 16 apart, identical x, y, z); an identical
        # 3-D point is one measurement, not a pixel conflict. Distinct returns on one ray remain conflicts.
        _, first = np.unique(P[:, :3], axis=0, return_index=True); P = P[np.sort(first)]
    out["lidar_points"] = {"nonzero": int(n_nonzero), "after_dedupe": int(len(P))}
    rng = np.linalg.norm(P[:, :3], axis=1)
    Pc = (S.R @ (S.C @ P[:, :3].T)).T + S.T                        # mqproj.py convention
    u, v, ok = project(Pc, S.K, S.D, S.wh)
    W = S.wh[0]
    V = dict(u=u[ok], v=v[ok], Z=Pc[ok, 2], rng=rng[ok], rel=np.zeros(ok.sum(), np.int8),
             absdt=np.zeros(ok.sum()), reason=np.zeros(ok.sum(), np.int8))
    V["lin"] = np.rint(V["v"]).astype(np.int64) * W + np.rint(V["u"]).astype(np.int64)
    if a.raw_conflict_stage == "projected":
        V["reason"][conflict_drop_all(V["lin"], np.ones(ok.sum(), bool))] = CONFLICT_PROJ
    out["versions"]["raw"] = V
    return out


def run_multiaqua(a):
    import multiprocessing as mp
    S = MultiAqua(a.zip)
    allf = S.frames()
    if a.frames.startswith("per_seq:"):
        n = int(a.frames.split(":")[1]); frames = []
        for sq in ("adr1", "bl1", "lj1", "lj2", "lj3", "lj4"):
            c = [f for f in allf if f.startswith(sq + "_")]
            frames += [c[i] for i in np.unique(np.linspace(0, len(c) - 1, min(n, len(c))).round().astype(int))]
    else:
        frames = [f for f in a.frames.split(",") if f]
    a.versions = ["raw"]; a.raw_conflict = "drop"
    outd = os.path.join(a.out, "multiaqua"); os.makedirs(f"{outd}/raw/depth", exist_ok=True)
    os.makedirs(f"{outd}/stats", exist_ok=True); os.makedirs(f"{outd}/cache", exist_ok=True)
    params = run_params(a); params["calibration"] = {"K": S.K.tolist(), "D": S.D.tolist(), "wh": S.wh}
    json.dump(params, open(f"{outd}/params.json", "w"), indent=1)
    hulls = hull_masks_from_labels(S, a, outd)
    params["hull_masks"] = {sq: int(m.sum()) for sq, m in hulls.items()}
    json.dump(params, open(f"{outd}/params.json", "w"), indent=1)
    print(f"[multiaqua] {len(frames)} frames; own-hull mask pixels per sequence {params['hull_masks']}", flush=True)
    pool = mp.get_context("spawn").Pool(a.workers, initializer=_init_mq, initargs=(a.zip, a))
    wm = WaterMask(a.water_model, grow=a.water_grow, device=a.device)
    todo = [f for f in frames if not os.path.exists(f"{outd}/stats/{f}.json")]
    done = len(frames) - len(todo); batch = 4 * a.workers
    def _results():
        for i in range(0, len(todo), batch):
            yield from pool.imap_unordered(mq_frame, todo[i:i + batch], chunksize=2)
    for k, res in enumerate(_results()):
        fr = res["frame"]; img = res.pop("image")
        if img is None:   # truncated / undecodable image in the release (e.g. lj1_3_010064.png): recorded, no GT
            json.dump({"frame": fr, "sequence": fr.split("_")[0], "errors": res["errors"] + ["image decode failed"],
                       "versions": {}}, open(f"{outd}/stats/{fr}.json", "w"))
            done += 1; continue
        water, grown, boats = wm(img)
        V = res["versions"]["raw"]
        st, depth, r = finish_version(V, grown, a, version="raw", hull=hulls.get(fr.split("_")[0]))
        st["tags"] = frame_tags(img, water, boats, V, r, a)
        rec = {"frame": fr, "sequence": fr.split("_")[0], "errors": res["errors"], "versions": {"raw": st},
               "lidar_points": res["lidar_points"],
               "water_frac_raw_mask": float(water.mean()), "water_frac_grown_mask": float(grown.mean())}
        write_depth_png(f"{outd}/raw/depth/{fr}.png", depth)
        if k % a.cache_every == 0: save_cache(f"{outd}/cache/{fr}_raw", V, r, img)
        json.dump(rec, open(f"{outd}/stats/{fr}.json", "w"))
        done += 1
        if done % 25 == 0: print(f"[multiaqua] {done}/{len(frames)}", flush=True)
    pool.close(); pool.join()


# --------------------------------------------------------------------------------------------------------- summarize
def turbo(x):
    x = np.clip(np.asarray(x, float), 0, 1)
    if x.size == 0: return np.zeros((0, 3), np.uint8)
    return cv2.applyColorMap((x * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)


REASON_BGR = {SGM_INVALID: (203, 192, 255), TAU: (0, 0, 255), WATER: (255, 128, 0), MIN_RANGE: (255, 0, 255),
              CONFLICT_PROJ: (0, 255, 255), CONFLICT: (0, 255, 255), KEPT: (0, 200, 0)}


def render(cache, title, max_depth, out_path):
    img = cv2.imread(cache + ".jpg"); c = np.load(cache + ".npz")
    u, v, Z, r = c["u"], c["v"], c["Z"], c["reason"]
    dim = (img * 0.55).astype(np.uint8)
    top = dim.copy(); bot = dim.copy()
    k = (r == KEPT) & (Z <= max_depth)
    col = turbo(Z[k] / max_depth)
    for (x, y), cc in zip(np.stack([u[k], v[k]], 1).astype(int), col): cv2.circle(top, (x, y), 2, tuple(int(t) for t in cc), -1)
    # bottom: removed points (<= max depth) by reason, kept in green; far points (> max depth) as small grey dots
    far = Z > max_depth
    for x, y in np.stack([u[far], v[far]], 1).astype(int)[::3]: cv2.circle(bot, (x, y), 1, (90, 90, 90), -1)
    for code in (KEPT, SGM_INVALID, WATER, MIN_RANGE, CONFLICT_PROJ, CONFLICT, TAU):
        m = (r == code) & ~far
        for x, y in np.stack([u[m], v[m]], 1).astype(int): cv2.circle(bot, (x, y), 2, REASON_BGR[code], -1)
    def label(im, lines):
        for i, (t, colr) in enumerate(lines):
            cv2.putText(im, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, t, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colr, 1, cv2.LINE_AA)
    label(top, [(title, (255, 255, 255)), (f"GT points (0-{max_depth:g} m, turbo: blue near, red far), n={int(k.sum())}", (255, 255, 255))])
    cnt = {code: int(((r == code) & ~far).sum()) for code in REASON_BGR}
    label(bot, [("removed <= %g m by reason / kept" % max_depth, (255, 255, 255)),
                (f"kept {cnt[KEPT]}", REASON_BGR[KEPT]), (f"stereo tau {cnt[TAU]}", REASON_BGR[TAU]),
                (f"SGM invalid {cnt[SGM_INVALID]}", REASON_BGR[SGM_INVALID]), (f"water {cnt[WATER]}", REASON_BGR[WATER]),
                (f"min range/hull {cnt[MIN_RANGE]}", REASON_BGR[MIN_RANGE]),
                (f"pixel conflict {cnt[CONFLICT_PROJ] + cnt[CONFLICT]}", REASON_BGR[CONFLICT]),
                ("dark grey dots: > %g m (any reason)" % max_depth, (160, 160, 160))])
    # colour bar
    bar = turbo(np.linspace(0, 1, 200)).reshape(1, 200, 3).repeat(10, 0)
    top[-18:-8, 8:208] = bar
    cv2.putText(top, f"0 m", (8, top.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(top, f"{max_depth:g} m", (180, top.shape[0] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.imwrite(out_path, np.vstack([top, bot]), [cv2.IMWRITE_JPEG_QUALITY, 82])


def select_frames(recs, ver, n=30, seed=0):
    """Stratified pick covering night, boats, overhead structure (bridges), near, far, open water, plus random."""
    rng = np.random.default_rng(seed)
    ok = [r for r in recs if r["versions"].get(ver)]
    T = {r["frame"]: r["versions"][ver]["tags"] for r in ok}
    P = {r["frame"]: r["versions"][ver]["pass_min_points"] for r in ok}
    cats = {
        "night": [f for f, t in T.items() if t["mean_brightness"] < 50],
        "boat": [f for f, t in T.items() if t["boat_instances"] > 0 and P[f]],
        "overhead_bridge": [f for f, t in T.items() if t["n_final_upper_third"] >= 20],
        "near_le10m": [f for f, t in T.items() if t["n_final_le10"] >= 20],
        "far_15_20m": [f for f, t in T.items() if t["n_final_15_20"] >= 50],
        "open_water_few_points": [f for f, t in T.items() if t["water_frac"] > 0.35 and not P[f]],
    }
    pick, why = [], {}
    per = max(1, n // (len(cats) + 1))
    for c, fs in cats.items():
        fs = [f for f in fs if f not in why]
        for f in rng.permutation(fs)[:per] if fs else []:
            pick.append(str(f)); why[str(f)] = c
    rest = [r["frame"] for r in ok if r["frame"] not in why]
    for f in rng.permutation(rest)[: n - len(pick)]:
        pick.append(str(f)); why[str(f)] = "random"
    return sorted(pick), why, {c: len(v) for c, v in cats.items()}


def aggregate_disp_bins(recs):
    out = []
    for i, (lo, hi) in enumerate(zip(DISP_BINS[:-1], DISP_BINS[1:])):
        n = sum(r["versions"]["uhrig"]["stereo_residual_by_disparity_bin"][i]["n"] for r in recs)
        sm = sum(r["versions"]["uhrig"]["stereo_residual_by_disparity_bin"][i]["sum"] for r in recs)
        no = sum(r["versions"]["uhrig"]["stereo_residual_by_disparity_bin"][i]["outliers"] for r in recs)
        meds = [r["versions"]["uhrig"]["stereo_residual_by_disparity_bin"][i]["median"] for r in recs
                if r["versions"]["uhrig"]["stereo_residual_by_disparity_bin"][i]["n"] >= 20]
        out.append({"d_lo_px": lo, "d_hi_px": hi, "n": n, "mean_residual_px": round(sm / n, 3) if n else None,
                    "median_of_frame_medians_px": round(float(np.median(meds)), 3) if meds else None,
                    "tau_outlier_rate": round(no / n, 4) if n else None})
    return out


def diag(a):
    """Compare Uhrig variants on the same frames (camera-time offsets, rotation source, gyro bias, SGM-invalid)."""
    res = {}
    for name in sorted(os.listdir(a.diag)):
        d = os.path.join(a.diag, name, a.seq, "stats")
        if not os.path.isdir(d): continue
        recs = [json.load(open(f"{d}/{f}")) for f in sorted(os.listdir(d))]
        ok = [r for r in recs if r["versions"].get("uhrig")]
        U = [r["versions"]["uhrig"] for r in ok]
        acc = {}
        for u in U:
            for j, x in u["by_scan_offset"].items():
                t = acc.setdefault(int(j), [0, 0]); t[0] += x["tau_outliers"]; t[1] += x["sgm_valid_le_max_depth_nonwater"]
        tot_o = sum(v[0] for v in acc.values()); tot_n = sum(v[1] for v in acc.values())
        res[name] = {"frames": len(ok), "frames_pass": sum(u["pass_min_points"] for u in U),
                     "final_points_le_max_depth": sum(u["n_final_le_max_depth"] for u in U),
                     "tau_outlier_rate_sgm_valid_le_max_depth": round(tot_o / max(tot_n, 1), 4), "n_sgm_valid": tot_n,
                     "tau_outlier_rate_by_scan_offset": {j: round(v[0] / max(v[1], 1), 4) for j, v in sorted(acc.items())},
                     "residual_by_disparity_bin": aggregate_disp_bins(ok)}
    json.dump(res, open(a.evidence, "w"), indent=1)
    for k, v in res.items():
        print(k, v["frames"], v["frames_pass"], v["final_points_le_max_depth"], v["tau_outlier_rate_sgm_valid_le_max_depth"],
              v["tau_outlier_rate_by_scan_offset"])


def summarize(a):
    os.makedirs(a.evidence, exist_ok=True)
    summary = {}
    for src in a.sources:
        d = os.path.join(a.out, src)
        recs = [json.load(open(f"{d}/stats/{f}")) for f in sorted(os.listdir(f"{d}/stats"))]
        params = json.load(open(f"{d}/params.json"))
        with open(f"{a.evidence}/per_frame_{src}.jsonl", "w") as fh:
            for r in recs: fh.write(json.dumps(r) + "\n")
        json.dump(params, open(f"{a.evidence}/params_{src}.json", "w"), indent=1)
        vers = [v for v in ("uhrig", "raw") if any(r["versions"].get(v) for r in recs)]
        S = {"n_frames": len(recs), "frames_with_errors": [[r["frame"], r["errors"]] for r in recs if r.get("errors")]}
        for ver in vers:
            ok = [r for r in recs if r["versions"].get(ver)]
            seqs = sorted({r["sequence"] for r in ok})
            per_seq = {}
            for sq in seqs + ["all"]:
                rs = [r for r in ok if sq == "all" or r["sequence"] == sq]
                tot = {}
                for r in rs:
                    for k, bc in r["versions"][ver]["stats_per_band"].items():
                        tot[k] = [x + y for x, y in zip(tot.get(k, [0] * 5), bc)]
                npass = sum(r["versions"][ver]["pass_min_points"] for r in rs)
                per_seq[sq] = {"frames": len(rs), "frames_pass_min_points": npass,
                               "frac_pass": round(npass / max(len(rs), 1), 4),
                               "frames_ge1_point": sum(r["versions"][ver]["n_final_le_max_depth"] > 0 for r in rs),
                               "points_per_band_all_frames": dict(zip(BAND_NAMES, tot.get("final_points", [0] * 5))),
                               "points_per_band_passing_frames": dict(zip(BAND_NAMES, [sum(r["versions"][ver]["stats_per_band"]["final_points"][i] for r in rs if r["versions"][ver]["pass_min_points"]) for i in range(5)])),
                               "step_totals_per_band": {k: dict(zip(BAND_NAMES, v)) for k, v in tot.items()},
                               "median_final_points_le_max_depth": float(np.median([r["versions"][ver]["n_final_le_max_depth"] for r in rs])) if rs else None}
                if any("n_final_if_conflict_stage_filtered" in r["versions"][ver] for r in rs):
                    per_seq[sq]["frames_pass_if_conflict_stage_filtered"] = sum(
                        r["versions"][ver].get("n_final_if_conflict_stage_filtered", 0) >= params["min_points"] for r in rs)
            S[ver] = per_seq
            if ver == "uhrig":
                acc = {}; h = np.zeros(40); sv = {b: [] for b in BAND_NAMES}; shifts = []; biases = []
                for r in ok:
                    u = r["versions"][ver]
                    for j, x in u["by_scan_offset"].items():
                        t = acc.setdefault(j, {k2: 0 for k2 in x})
                        for k2 in x: t[k2] += x[k2]
                    h += np.array(u["stereo_residual_px_le_max_depth_nonwater"]["hist"])
                    for b in BAND_NAMES:
                        if u["sgm_valid_rate_at_lidar"][b] is not None: sv[b].append(u["sgm_valid_rate_at_lidar"][b])
                    s = u["motion"]["centre_scan_shift_px_le_max_depth"]
                    if s["median"] is not None: shifts.append(s["median"])
                    biases.append(u["motion"]["gyro_bias_rad_s"])
                for j in acc:
                    acc[j]["tau_outlier_rate"] = round(acc[j]["tau_outliers"] / max(acc[j]["sgm_valid_le_max_depth_nonwater"], 1), 4)
                    acc[j]["sgm_valid_rate"] = round(acc[j]["sgm_valid_le_max_depth_nonwater"] / max(acc[j]["projected_le_max_depth"], 1), 4)
                S["uhrig_stereo_residual_by_disparity_bin"] = aggregate_disp_bins(ok)
                S["uhrig_alignment_by_scan_offset"] = dict(sorted(acc.items(), key=lambda kv: int(kv[0])))
                S["uhrig_stereo_residual_hist_px"] = {"edges": np.linspace(-10, 10, 41).tolist(), "counts": h.astype(int).tolist()}
                S["uhrig_sgm_valid_rate_at_lidar_median_over_frames"] = {b: (float(np.median(x)) if x else None) for b, x in sv.items()}
                S["uhrig_deskew_shift_px_median_over_frames"] = {"median": float(np.median(shifts)) if shifts else None,
                                                                "p95": float(np.percentile(shifts, 95)) if shifts else None}
                S["gyro_bias_deg_s_median"] = np.degrees(np.median(np.array(biases), 0)).round(3).tolist() if biases else None
            if src != "multiaqua":
                cs = [r["centre_scan"] for r in recs if "centre_scan" in r]
                S["timing_ms"] = {k: {"median": float(np.median([c[k] for c in cs])), "min": float(np.min([c[k] for c in cs])),
                                      "max": float(np.max([c[k] for c in cs]))} for k in ("file_ts_minus_cam_ms", "fov_time_minus_cam_ms")}
        # renders: the same frames for every version of a source
        pick, why, cat_sizes = select_frames(recs, vers[0], a.n_renders)
        S["render_selection"] = {"frames": {f: why[f] for f in pick}, "category_sizes": cat_sizes}
        rd = f"{a.evidence}/renders_{src}"; os.makedirs(rd, exist_ok=True)
        R = {r["frame"]: r for r in recs}
        for ver in vers:
            for f in pick:
                st = R[f]["versions"][ver]
                if not st: continue
                title = (f"{src} {f} {ver} [{why[f]}] final<= {params['max_depth']:g} m: {st['n_final_le_max_depth']}"
                         f" ({'pass' if st['pass_min_points'] else 'FAIL'} >= {params['min_points']})")
                render(f"{d}/cache/{f}_{ver}", title, params["max_depth"], f"{rd}/{f}_{ver}.jpg")
        summary[src] = S
    json.dump(summary, open(f"{a.evidence}/coverage_summary.json", "w"), indent=1)
    print(json.dumps({s: {v: summary[s][v]["all"] for v in ("uhrig", "raw") if v in summary[s]} for s in summary}, indent=1))


# -------------------------------------------------------------------------------------------------------------- CLI
def parse_frames(spec, n):
    if spec.startswith("every:"):
        return list(range(0, n, int(spec.split(":")[1])))
    return [int(x) for x in spec.split(",") if x]


def run_params(a):
    return {k: v for k, v in vars(a).items() if not k.startswith("_") and k != "func"}


def common_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--water-model", required=True)
    p.add_argument("--water-grow", type=int, default=10, help="D11: grow the water mask by this many px (square kernel)")
    p.add_argument("--water-filter", choices=["off", "remove"], default="off",
                   help="off (default, unadapted GT): points on the water mask are kept and only tagged; remove: D11 adaptation")
    p.add_argument("--device", default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cv-threads", type=int, default=1)
    p.add_argument("--max-depth", type=float, default=20.0, help="scope (README): keep GT depth <= this")
    p.add_argument("--min-range", type=float, default=1.0,
                   help="step 6 minimum LiDAR range (m, sensor-frame range). README: 'each LiDAR's minimum range "
                        "(from its spec)'; the LiDAR models' spec values were not verified -> 1.0 m (conservative)")
    p.add_argument("--min-points", type=int, default=50, help="D10 frame filter")
    p.add_argument("--cache-every", type=int, default=1, help="write the render cache (jpg + npz) for every Nth frame")
    p.add_argument("--raw-conflict-stage", choices=["projected", "filtered"], default="projected",
                   help="raw GT: resolve per-pixel conflicts among all projected points (conservative, default) or "
                        "among points left after water / range")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pohang"); common_args(p)
    p.add_argument("--raw-dir", required=True); p.add_argument("--seq", required=True)
    p.add_argument("--frames", default="every:50", help="'every:N' or comma list of stereo frame indices")
    p.add_argument("--shard", default=None, help="i/n")
    p.add_argument("--versions", default="uhrig,raw", type=lambda s: s.split(","))
    p.add_argument("--n-scans", type=int, default=11, help="Uhrig: accumulate 11 scans (centre +- 5)")
    p.add_argument("--rotation-source", choices=["imu", "slam"], default="imu",
                   help="rotation between point time and image time: LiDAR-IMU gyro (README step 1, default) or SLAM slerp")
    p.add_argument("--deskew-translation", choices=["slam", "none"], default="slam",
                   help="translation between point time and image time from the SLAM trajectory (linear interpolation)")
    p.add_argument("--gyro-bias-window-s", type=float, default=60.0,
                   help="gyro bias = mean(gyro - SLAM rate) over this centred window; 0 = no bias correction")
    p.add_argument("--gt-frame", choices=["undistorted", "distorted"], default="undistorted",
                   help="README step 3: GT in the undistorted left camera (same K); 'distorted' = released image")
    p.add_argument("--rect-alpha", type=float, default=1.0)
    p.add_argument("--cam-time-offset-s", type=float, default=0.0,
                   help="added to stereo/timestamp.txt times (diagnostic only; the G1 IMU clock check decides any shift)")
    p.add_argument("--stereo-disp-offset-px", type=float, default=0.0,
                   help="add a constant to SGM disparity before the tau check (G1: released stereo calibration offset)")
    p.add_argument("--tau-px", type=float, default=3.0); p.add_argument("--tau-rel", type=float, default=0.05)
    p.add_argument("--sgm-invalid", choices=["drop", "keep"], default="drop",
                   help="LiDAR points with no valid SGM disparity: drop (cannot be verified; conservative) or keep")
    p.add_argument("--sgm-num-disp", type=int, default=768, help="covers depth >= fB/768 (~1.9 m)")
    p.add_argument("--sgm-block", type=int, default=5)
    p.add_argument("--sgm-mode", choices=["sgbm", "hh", "3way"], default="sgbm")
    p.add_argument("--sgm-uniqueness", type=int, default=10)
    p.add_argument("--sgm-lr-max-diff", type=int, default=1)
    p.add_argument("--sgm-speckle-window", type=int, default=100)
    p.add_argument("--sgm-speckle-range", type=int, default=2)
    p.set_defaults(func=run_pohang, raw_conflict="closest_time")
    p = sp.add_parser("multiaqua"); common_args(p)
    p.add_argument("--zip", required=True)
    p.add_argument("--frames", default="per_seq:50", help="'per_seq:N' (evenly spaced per sequence) or comma list")
    p.add_argument("--hull-label-freq", type=float, default=0.9,
                   help="own-hull mask: pixels unlabelled in >= this fraction of a sequence's labelled frames; 0 = off")
    p.add_argument("--no-mq-dedupe", dest="mq_dedupe", action="store_false",
                   help="keep exact duplicate x,y,z rows (default: collapse them before the conflict rule)")
    p.set_defaults(func=run_multiaqua, raw_conflict="drop", versions=["raw"])
    p = sp.add_parser("summarize")
    p.add_argument("--out", required=True); p.add_argument("--evidence", required=True)
    p.add_argument("--sources", default="pohang00,multiaqua", type=lambda s: s.split(","))
    p.add_argument("--n-renders", type=int, default=30)
    p.set_defaults(func=summarize)
    p = sp.add_parser("diag")
    p.add_argument("--diag", required=True); p.add_argument("--seq", default="pohang00")
    p.add_argument("--evidence", required=True, help="output JSON path")
    p.set_defaults(func=diag)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
