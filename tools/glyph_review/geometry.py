"""Pixel-space measurements and bounded normal-displacement calibration.

Widths are geometric proxies, not measurements of physical pen pressure.
All decisions use the quantized image that will actually be saved.
"""
from dataclasses import dataclass
import inspect

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, label, map_coordinates
from skimage.morphology import medial_axis


@dataclass(frozen=True)
class Settings:
    threshold: float = 0.5
    seed: int = 0
    min_area: int = 1
    max_shift: float = 2.0
    strength: float = 1.0

    def __post_init__(self):
        if not 0 < self.threshold < 1 or not 0 <= self.strength <= 1:
            raise ValueError('Expected threshold in (0,1) and strength in [0,1]')
        if not np.isfinite(self.max_shift) or self.max_shift <= 0 or self.min_area < 1:
            raise ValueError('max_shift must be positive and min_area >= 1')


def gray(image):
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError('Expected uint8 L or RGB image')
    if image.ndim == 3 and image.shape[2] != 3:
        raise ValueError('Expected RGB, not RGBA')
    return np.asarray(Image.fromarray(image).convert('L'), dtype=np.float32) / 255


def topology(mask, min_area=1):
    """Dual connectivity: 8-connected ink, 4-connected background; padded outside."""
    foreground, _ = label(mask, structure=np.ones((3, 3)))
    padded = np.pad(~mask, 1, constant_values=True)
    background, _ = label(padded)  # 4-connectivity
    outside = background[0, 0]
    holes = background[1:-1, 1:-1].copy()
    holes[(holes == outside) | mask] = 0

    def significant(labels):
        counts = np.bincount(labels.ravel())
        ids = np.flatnonzero(counts >= min_area)
        return ids[ids != 0]

    return foreground, significant(foreground), holes, significant(holes)


def _one_to_one(a, ids_a, b, ids_b):
    """Equal counts alone miss simultaneous splits, merges and component loss."""
    if len(ids_a) != len(ids_b):
        return False
    if not len(ids_a):
        return True
    overlap = np.isin(a, ids_a) & np.isin(b, ids_b)
    radix = int(b.max()) + 1
    pairs = np.unique(a[overlap].astype(np.int64)*radix + b[overlap])
    aa, bb = pairs // radix, pairs % radix
    return (len(pairs) == len(ids_a) and
            np.array_equal(np.sort(aa), np.sort(ids_a)) and
            np.array_equal(np.sort(bb), np.sort(ids_b)))


def topology_preserved(original, candidate, min_area=1):
    a, ia, h, ih = topology(original, min_area)
    b, ib, k, ik = topology(candidate, min_area)
    return _one_to_one(a, ia, b, ib) and _one_to_one(h, ih, k, ik)


def measure(image, settings=Settings()):
    luminance = gray(image)
    mask = (1-luminance) >= settings.threshold
    distance = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    seed_arg = 'rng' if 'rng' in inspect.signature(medial_axis).parameters else 'random_state'
    skeleton = medial_axis(mask, **{seed_arg: settings.seed}) if mask.any() else mask.copy()
    widths = 2*distance[skeleton]
    q = np.percentile(widths, [10, 50, 90]) if widths.size else np.zeros(3)
    _, components, _, holes = topology(mask, settings.min_area)
    yy, xx = np.where(mask)
    bbox_area = int((yy.max()-yy.min()+1)*(xx.max()-xx.min()+1)) if yy.size else 0
    stats = dict(p10=float(q[0]), p50=float(q[1]), p90=float(q[2]),
                 width_contrast=float(q[2]/q[0]) if q[0] else None,
                 ink_area=int(mask.sum()), soft_ink=float((1-luminance).sum()),
                 skeleton_pixels=int(skeleton.sum()),
                 equivalent_width=float(mask.sum()/max(1, skeleton.sum())),
                 normalized_p50=float(q[1]/np.sqrt(bbox_area)) if bbox_area else 0.,
                 components=len(components), holes=len(holes), valid=bool(widths.size and mask.sum() >= 25))
    return stats, mask, skeleton, distance


def displace(image, shift, settings=Settings()):
    """Warp original grayscale/RGB near contours; never redraw a binary mask.

    Positive shifts thicken. Zero is an exact identity, including grey texture.
    This is an approximate normal warp, not a guarantee of target width.
    """
    field = np.broadcast_to(np.asarray(shift, dtype=np.float32), image.shape[:2])
    if not np.isfinite(field).all():
        raise ValueError('Non-finite displacement')
    field = np.clip(field, -settings.max_shift, settings.max_shift)
    if not np.any(field):
        return image.copy()
    mask = (1-gray(image)) >= settings.threshold
    if not mask.any():
        return image.copy()
    padded = np.pad(mask, 1)
    phi = (distance_transform_edt(padded)-distance_transform_edt(~padded))[1:-1, 1:-1]
    gy, gx = np.gradient(gaussian_filter(phi, .8))
    norm = np.maximum(np.hypot(gy, gx), 1e-6)
    y, x = np.indices(mask.shape, dtype=np.float32)
    coords = [y + field*gy/norm, x + field*gx/norm]
    band = np.abs(phi) <= np.abs(field)+2
    channels = image[..., None] if image.ndim == 2 else image
    result = image.copy()
    target = result[..., None] if result.ndim == 2 else result
    for i in range(channels.shape[2]):
        warped = map_coordinates(channels[..., i].astype(np.float32), coords,
                                 order=1, mode='constant', cval=255., prefilter=False)
        target[..., i][band] = np.clip(np.rint(warped[band]), 0, 255).astype(np.uint8)
    return result


def width_error(stats, target):
    return float(np.mean([abs(np.log(max(stats[k], .5)/max(float(v), .5)))
                          for k, v in target.items()]))


def calibrate(image, target=None, delta=None, method='uniform', settings=Settings()):
    if (target is None) == (delta is None):
        raise ValueError('Specify exactly one of target or delta')
    if method not in ('uniform', 'profile'):
        raise ValueError(method)
    if delta is not None and (not np.isfinite(delta) or abs(delta) > settings.max_shift):
        raise ValueError('Manual delta exceeds max_shift or is non-finite')
    if target is not None:
        if not target or set(target)-{'p10', 'p50', 'p90'} or 'p50' not in target:
            raise ValueError('Target requires p50; optional p10/p90')
        values = [float(target[k]) for k in ('p10','p50','p90') if k in target]
        if not all(np.isfinite(v) and v > 0 for v in values) or values != sorted(values):
            raise ValueError('Target widths must be positive, finite and ordered')
        if method == 'profile' and set(target) != {'p10', 'p50', 'p90'}:
            raise ValueError('Profile calibration requires all three target quantiles')
    before, mask, skeleton, distance = measure(image, settings)
    baseline = width_error(before, target) if target else None
    record = dict(before=before, after=before.copy(), target=target, requested_delta=delta,
                  effective_shift_max=0., selected_gain=0., objective_before=baseline,
                  objective_after=baseline, topology_rejected=0, changed=False, status='unchanged')
    if settings.strength == 0 or not before['valid']:
        record['status'] = 'zero_strength' if settings.strength == 0 else 'empty_input'
        return image.copy(), record
    if delta is not None:
        field = np.full(mask.shape, delta*settings.strength, dtype=np.float32)
        gains = [1., .75, .5, .25]
    elif method == 'uniform':
        desired = (float(target['p50'])-before['p50'])/2
        maximum = settings.max_shift*settings.strength
        choices = np.unique(np.r_[np.linspace(-maximum, maximum, 9), np.clip(desired*settings.strength, -maximum, maximum)])
        # Evaluate smaller moves first so ties never favor unnecessary deformation.
        gains = sorted(choices, key=abs)
        field = np.ones(mask.shape, dtype=np.float32)
    else:
        keys = [k for k in ('p10','p50','p90') if k in target]
        src = np.array([before[k] for k in keys])
        dst = np.array([target[k] for k in keys])
        unique = np.unique(src)
        dst = np.array([dst[src == v].mean() for v in unique])
        widths = 2*distance[skeleton]
        shifts = (np.interp(widths, unique, dst)-widths)/2
        local = np.zeros(mask.shape, dtype=np.float32)
        local[skeleton] = np.clip(shifts, -settings.max_shift, settings.max_shift)
        _, nearest = distance_transform_edt(~skeleton, return_indices=True)
        field = gaussian_filter(local[tuple(nearest)], 1.)*settings.strength
        gains = [.25, .5, .75, 1.]
    best = image.copy()
    best_error = baseline
    for gain in gains:
        if gain == 0:
            continue
        shift = field*gain
        candidate = displace(image, shift, settings)
        after, new_mask, _, _ = measure(candidate, settings)
        if not topology_preserved(mask, new_mask, settings.min_area):
            record['topology_rejected'] += 1
            continue
        if not after['valid'] or np.array_equal(candidate, image):
            continue
        error = width_error(after, target) if target else None
        if target and not error < best_error-1e-8:
            continue
        best, best_error = candidate, error
        record.update(after=after, effective_shift_max=float(np.abs(shift).max()),
                      selected_gain=float(gain), objective_after=error, changed=True, status='adjusted')
        if delta is not None:  # Largest safe manual request, without pretending to hit a target.
            break
    if not record['changed']:
        record['status'] = 'no_safe_improvement' if target else 'manual_rejected_or_no_change'
    return best, record
