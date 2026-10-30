"""to_lars_rgb: our (class index, segment id) panoptic maps -> LaRS evaluator PNG encoding."""

import numpy as np

from water_debris_seg.lars_panoptic_png import LARS_CATEGORY_IDS, NUM_LARS, to_lars_rgb


def _segments(rgb):
    ids = rgb[..., 0].astype(np.int64) * 65536 + rgb[..., 1].astype(np.int64) * 256 + rgb[..., 2]
    return ids


def test_classes_instances_and_void():
    cls = np.full((4, 6), NUM_LARS)                  # void everywhere
    seg = np.full((4, 6), -1)
    cls[0, :] = 2; seg[0, :] = 100002                # sky (stuff)
    cls[1, :] = 1; seg[1, :] = 7                     # water
    cls[2, :3] = 3; seg[2, :3] = 0                   # boat, segment 0
    cls[2, 3:] = 3; seg[2, 3:] = 1                   # second boat
    cls[3, :2] = 10; seg[3, :2] = 0                  # "Other" sharing segment id 0 with the first boat
    rgb = to_lars_rgb(np.stack([cls, seg], -1))
    assert rgb.dtype == np.uint8
    assert (rgb[0, :, 0] == 5).all() and (rgb[1, :, 0] == 3).all()
    assert (rgb[2, :, 0] == LARS_CATEGORY_IDS[3]).all() and (rgb[3, :2, 0] == 19).all()
    assert (rgb[3, 2:] == 0).all()                   # void -> (0, 0, 0)
    ids = _segments(rgb)
    assert ids[2, 0] != ids[2, 5]                    # two boats stay two instances
    assert len(np.unique(ids[ids > 0])) == 5         # sky, water, 2 boats, other
    assert (rgb[..., 1:].sum(-1)[cls < NUM_LARS] > 0).all()   # every non-void pixel has an instance id


def test_torch_input():
    import torch

    preds = torch.stack([torch.tensor([[0, 11]]), torch.tensor([[100000, -1]])], -1)
    rgb = to_lars_rgb(preds)
    assert rgb[0, 0, 0] == 1 and (rgb[0, 1] == 0).all()
