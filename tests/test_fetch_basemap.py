"""Geometry checks for dashboard/fetch_basemap.py using a tiny hand-made map fixture.

The fixture is plain geometry (no results, no scores); it only exercises clipping,
simplification, ring stitching and encoding so the offline basemap pipeline is trusted.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_basemap", ROOT / "dashboard" / "fetch_basemap.py")
fb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fb)

BBOX = (38.88, -77.06, 38.92, -77.00)


def pt(lat, lon):
    return {"lat": lat, "lon": lon}


def decode(flat):
    la, lo = flat[0], flat[1]
    out = [(la / fb.SCALE, lo / fb.SCALE)]
    for i in range(2, len(flat), 2):
        la += flat[i]
        lo += flat[i + 1]
        out.append((la / fb.SCALE, lo / fb.SCALE))
    return out


def fixture():
    return {"elements": [
        # long avenue that leaves the box on both sides, with collinear extra points
        {"type": "way", "tags": {"highway": "primary"}, "geometry": [
            pt(38.90, -77.20), pt(38.90, -77.04), pt(38.90, -77.03), pt(38.90, -76.90)]},
        {"type": "way", "tags": {"highway": "residential"}, "geometry": [pt(38.89, -77.05), pt(38.91, -77.05)]},
        {"type": "way", "tags": {"highway": "footway"}, "geometry": [pt(38.89, -77.04), pt(38.91, -77.04)]},
        # closed park way
        {"type": "way", "tags": {"leisure": "park", "name": "Test Green"}, "geometry": [
            pt(38.895, -77.045), pt(38.895, -77.035), pt(38.905, -77.035), pt(38.905, -77.045), pt(38.895, -77.045)]},
        # water relation made of two open outer ways that must be stitched into one ring
        {"type": "relation", "tags": {"natural": "water", "name": "Test River"}, "members": [
            {"type": "way", "role": "outer", "geometry": [pt(38.881, -77.059), pt(38.881, -77.001), pt(38.884, -77.001)]},
            {"type": "way", "role": "outer", "geometry": [pt(38.884, -77.001), pt(38.884, -77.059), pt(38.881, -77.059)]},
        ]},
    ]}


def test_stitch_joins_fragments_into_one_ring():
    ways = [[(0, 0), (1, 0), (1, 1)], [(1, 1), (0, 1), (0, 0)]]
    rings = fb.stitch(ways)
    assert len(rings) == 1 and len(rings[0]) == 4


def test_clip_polyline_cuts_at_window_edge():
    win = (-100.0, -100.0, 100.0, 100.0)
    runs = fb.clip_polyline([(-500.0, 0.0), (500.0, 0.0)], win)
    assert len(runs) == 1
    assert runs[0][0] == (-100.0, 0.0) and runs[0][-1] == (100.0, 0.0)


def test_clip_polygon_stays_inside_window():
    win = (0.0, 0.0, 10.0, 10.0)
    ring = [(-5.0, -5.0), (15.0, -5.0), (15.0, 15.0), (-5.0, 15.0)]
    out = fb.clip_polygon(ring, win)
    assert all(0.0 <= x <= 10.0 and 0.0 <= y <= 10.0 for x, y in out)
    assert abs(fb.area(out) - 100.0) < 1e-6


def test_rdp_removes_collinear_points_only():
    pts = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 2.0)]
    out = fb.rdp(pts, 0.1)
    assert (1.0, 0.0) not in out or (2.0, 0.0) in out
    assert out[0] == pts[0] and out[-1] == pts[-1]


def test_process_classifies_clips_and_encodes():
    out = fb.process(fixture(), BBOX)
    assert len(out["roads"]["major"]) == 1
    assert len(out["roads"]["minor"]) == 1
    assert out["roads"]["mid"] == []                      # footway is not drawn
    major = decode(out["roads"]["major"][0])
    lons = [p[1] for p in major]
    assert min(lons) >= BBOX[1] - 1e-5 and max(lons) <= BBOX[3] + 1e-5   # clipped to the box
    assert len(major) == 2                                 # collinear points were simplified away
    assert len(out["parks"]) == 1 and len(out["water"]) == 1
    assert out["bbox"] == [round(v, 5) for v in BBOX]


def test_labels_only_for_large_named_features():
    out = fb.process(fixture(), BBOX)
    names = {l["t"] for l in out["labels"]}
    assert "Test River" in names            # about 6 km x 330 m, well over the water label threshold
    for l in out["labels"]:
        assert BBOX[0] <= l["lat"] <= BBOX[2] and BBOX[1] <= l["lon"] <= BBOX[3]


def test_output_has_no_urls():
    import json
    text = json.dumps(fb.process(fixture(), BBOX))
    assert "http" not in text