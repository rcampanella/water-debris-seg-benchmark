"""Synthetic unit tests for build_depth_gt.py. Run: python test_build_depth_gt.py  (or pytest)."""
import os, sys, tempfile
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_depth_gt as g


def test_quat_order():
    # [x, y, z, w]: (0.5, 0.5, 0.5, 0.5) maps camera (x right, y down, z fwd) to FRD body (z -> x, x -> y, y -> z)
    R = g.quat2R([0.5, 0.5, 0.5, 0.5])
    assert np.allclose(R @ [0, 0, 1], [1, 0, 0]) and np.allclose(R @ [1, 0, 0], [0, 1, 0])
    assert np.allclose(g.quat2R([0, 0, 0, 1]), np.eye(3))


def test_projection_roundtrip():
    rng = np.random.default_rng(0)
    K = np.array([[1800., 0, 1020], [0, 1800., 530], [0, 0, 1]]); wh = (2048, 1080)
    D = np.array([-0.08, 0.10, -0.003, 0.0008, 0.09])
    uv = rng.uniform([0, 0], [2047, 1079], (500, 2)); Z = rng.uniform(1, 40, 500)
    # pinhole: back-project, project, compare
    P = np.c_[(uv[:, 0] - K[0, 2]) / K[0, 0] * Z, (uv[:, 1] - K[1, 2]) / K[1, 1] * Z, Z]
    u, v, ok = g.project(P, K, None, wh)
    assert ok.all() and np.allclose(u, uv[:, 0]) and np.allclose(v, uv[:, 1])
    # distorted: undistortPoints is the inverse of projectPoints
    n = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    P = np.c_[n * Z[:, None], Z]
    u, v, ok = g.project(P, K, D, wh)
    assert np.abs(u[ok] - uv[ok, 0]).max() < 0.05 and np.abs(v[ok] - uv[ok, 1]).max() < 0.05 and ok.mean() > 0.98
    # behind camera / outside image -> not ok
    u, v, ok = g.project(np.array([[0, 0, -5.], [100, 0, 1.], [0, 0, 5.]]), K, None, wh)
    assert ok.tolist() == [False, False, True]


def test_lidar_camera_chain():
    # point in LiDAR frame -> camera -> back: R_cl, t_cl composition as in Pohang.__init__
    R_bl, t_bl = g.quat2R([0.998, -0.05, 0.03, -0.005]), np.array([5.2, -0.06, -0.39])
    R_bc, t_bc = g.quat2R([0.505, 0.493, 0.503, 0.499]), np.array([5.0, -0.44, -1.77])
    R_cl = R_bc.T @ R_bl; t_cl = R_bc.T @ (t_bl - t_bc)
    p = np.array([10., 2., -1.]); pc = R_cl @ p + t_cl
    pb = R_bl @ p + t_bl
    assert np.allclose(pc, R_bc.T @ (pb - t_bc))
    assert np.allclose(R_cl.T @ (pc - t_cl), p)


def test_tau_rule():
    dl = np.array([100., 100., 100., 40., 40., 200.])
    ds = np.array([103., 96., 102., 42.5, 37., 191.])
    # |e|: 3 (3%, below 5%) ; 4 (4%) ; 2 ; 2.5 (6.25% but < 3px) ; 3 (7.5%) ; 9 (4.5%)
    assert g.tau_outlier(dl, ds).tolist() == [False, False, False, False, True, False]
    assert g.tau_outlier(np.array([60.]), np.array([57.])).tolist() == [True]   # 3 px and exactly 5%


def test_png_roundtrip():
    d = np.zeros((7, 9), np.float32); d[1, 2] = 0.004; d[3, 4] = 19.99; d[6, 8] = 5.123; d[0, 0] = np.nan
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "x.png"); g.write_depth_png(p, d)
        a = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    assert a.dtype == np.uint16 and a.shape == d.shape
    assert a[0, 0] == 0 and a[1, 2] == 1 and a[3, 4] == round(19.99 * 256) and a[6, 8] == round(5.123 * 256)
    back = g.decode_depth_png(a)
    assert np.abs(back[d > 0] - np.nan_to_num(d)[d > 0]).max() <= 0.5 / 256 + 1e-6
    try:
        g.encode_depth_png(np.array([300.0])); assert False
    except ValueError:
        pass


def test_conflict_rules():
    lin = np.array([5, 5, 7, 9, 9, 9, 11])
    cand = np.ones(7, bool)
    assert g.conflict_drop_all(lin, cand).tolist() == [True, True, False, True, True, True, False]
    cand2 = cand.copy(); cand2[0] = False          # non-candidates neither count nor get removed
    assert g.conflict_drop_all(lin, cand2).tolist() == [False, False, False, True, True, True, False]
    dt = np.array([0.03, 0.01, 0.5, 0.02, 0.02, 0.04, 0.0])
    # pixel 5 keeps the later-listed but closer-in-time point; pixel 9 has a tie at the minimum -> all dropped
    assert g.conflict_closest_time(lin, dt, cand).tolist() == [True, False, False, True, True, True, False]
    # uhrig: fB = 1000; pixel 5 depths 10 m / 10.2 m -> disparities 100 / 98 (2 px) agree -> closest time kept;
    # pixel 9 depths 10 / 12 / 10 -> 100 vs 83 disagree -> dropped
    Z = np.array([10., 10.2, 5., 10., 12., 10., 3.]); dt2 = np.array([0.03, 0.01, 0.5, 0.02, 0.05, 0.04, 0.0])
    assert g.conflict_uhrig(lin, Z, dt2, cand, 1000.).tolist() == [True, False, False, True, True, True, False]


def test_band_counts():
    Z = np.array([0.5, 5.0, 5.01, 10, 14.9, 15.0, 20.0, 20.01, 80])
    assert g.band_counts(Z) == [2, 2, 2, 1, 2]   # (0,5] (5,10] (10,15] (15,20] >20


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
