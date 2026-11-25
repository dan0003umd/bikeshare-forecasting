#!/usr/bin/env python3
"""Build the offline street basemap used by the dashboard map (run once, needs internet).

    python dashboard/fetch_basemap.py

Uses only the Python standard library. It downloads street, water and park outlines for the
area around the 50 stations from OpenStreetMap (Overpass API), clips and simplifies them, and
writes dashboard/basemap_dc.json. build_dashboard.py embeds that file into index.html, so the
finished dashboard is fully offline. Map data (c) OpenStreetMap contributors, ODbL.

If the download is blocked on your network:
    python dashboard/fetch_basemap.py --print-query
open the printed address in a browser, save the response as dashboard/overpass_raw.json, then
    python dashboard/fetch_basemap.py --from-file dashboard/overpass_raw.json
"""
import argparse
import csv
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
MAJOR = {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}
MID = {"secondary", "tertiary", "secondary_link", "tertiary_link"}
MINOR = {"residential", "unclassified", "living_street"}
ROAD_RE = "^(" + "|".join(sorted(MAJOR | MID | MINOR)) + ")$"
ROAD_TOL = {"major": 2.0, "mid": 2.5, "minor": 3.0}   # simplification tolerance, meters
POLY_TOL = 4.0
MIN_WATER_AREA = 800.0       # square meters, after clipping
MIN_PARK_AREA = 2500.0
MIN_LABEL_AREA = {"water": 20000.0, "park": 40000.0}
MAX_LABELS = 14
SCALE = 1e5                  # stored coordinates are integer multiples of 1e-5 degrees (~1 m)


# ---------------------------------------------------------------- geometry helpers
class Frame:
    """Local meter frame around the area, used only for clipping and simplification."""

    def __init__(self, bbox):
        s, w, n, e = bbox
        self.lat0 = (s + n) / 2.0
        self.lon0 = (w + e) / 2.0
        self.mlat = 110574.0
        self.mlon = 111320.0 * math.cos(math.radians(self.lat0))

    def to_xy(self, lat, lon):
        return ((lon - self.lon0) * self.mlon, (lat - self.lat0) * self.mlat)

    def to_ll(self, x, y):
        return (self.lat0 + y / self.mlat, self.lon0 + x / self.mlon)


def clip_segment(p0, p1, win):
    """Liang-Barsky. Returns the clipped (start, end) or None if fully outside."""
    x0, y0 = p0
    x1, y1 = p1
    xmin, ymin, xmax, ymax = win
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)):
        if p == 0:
            if q < 0:
                return None
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
    return (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)


def inside_window(pts, win):
    xmin, ymin, xmax, ymax = win
    return all(xmin <= x <= xmax and ymin <= y <= ymax for x, y in pts)


def clip_polyline(pts, win):
    """Return a list of polylines (each at least 2 points) that lie inside the window."""
    if len(pts) < 2:
        return []
    if inside_window(pts, win):
        return [list(pts)]
    runs, cur = [], []
    for a, b in zip(pts, pts[1:]):
        seg = clip_segment(a, b, win)
        if seg is None:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
            continue
        s, e = seg
        if cur and (abs(cur[-1][0] - s[0]) > 1e-6 or abs(cur[-1][1] - s[1]) > 1e-6):
            if len(cur) > 1:
                runs.append(cur)
            cur = []
        if not cur:
            cur = [s]
        cur.append(e)
    if len(cur) > 1:
        runs.append(cur)
    return runs


def clip_polygon(ring, win):
    """Sutherland-Hodgman clip of a ring (no repeated closing point) to the window."""
    xmin, ymin, xmax, ymax = win
    if inside_window(ring, win):
        return list(ring)

    def run(poly, inside, cross):
        out = []
        if not poly:
            return out
        prev = poly[-1]
        for cur in poly:
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
            prev = cur
        return out

    def at_x(xv):
        return lambda p, c: (xv, p[1] + (xv - p[0]) / (c[0] - p[0]) * (c[1] - p[1]))

    def at_y(yv):
        return lambda p, c: (p[0] + (yv - p[1]) / (c[1] - p[1]) * (c[0] - p[0]), yv)

    poly = list(ring)
    poly = run(poly, lambda p: p[0] >= xmin, at_x(xmin))
    poly = run(poly, lambda p: p[0] <= xmax, at_x(xmax))
    poly = run(poly, lambda p: p[1] >= ymin, at_y(ymin))
    poly = run(poly, lambda p: p[1] <= ymax, at_y(ymax))
    return poly


def rdp(pts, tol):
    """Iterative Douglas-Peucker simplification."""
    n = len(pts)
    if n < 3:
        return list(pts)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        ax, ay = pts[i]
        bx, by = pts[j]
        dx, dy = bx - ax, by - ay
        l2 = dx * dx + dy * dy
        dmax, idx = -1.0, -1
        for k in range(i + 1, j):
            px, py = pts[k]
            if l2 == 0:
                d = math.hypot(px - ax, py - ay)
            else:
                t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
                d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
            if d > dmax:
                dmax, idx = d, k
        if dmax > tol:
            keep[idx] = True
            stack.append((i, idx))
            stack.append((idx, j))
    return [p for p, k in zip(pts, keep) if k]


def area(ring):
    s = 0.0
    for (x0, y0), (x1, y1) in zip(ring, ring[1:] + ring[:1]):
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def stitch(ways):
    """Join open way fragments (lists of (lon, lat)) into rings by matching end points."""
    ways = [list(w) for w in ways if len(w) >= 2]
    rings = []
    while ways:
        cur = ways.pop(0)
        changed = True
        while changed and cur[0] != cur[-1]:
            changed = False
            for i, w in enumerate(ways):
                if w[0] == cur[-1]:
                    cur = cur + w[1:]
                elif w[-1] == cur[-1]:
                    cur = cur + w[::-1][1:]
                elif w[-1] == cur[0]:
                    cur = w[:-1] + cur
                elif w[0] == cur[0]:
                    cur = w[::-1][:-1] + cur
                else:
                    continue
                ways.pop(i)
                changed = True
                break
        if cur[0] == cur[-1] and len(cur) >= 4:
            rings.append(cur[:-1])
    return rings


def encode(points_ll):
    """Delta-encode (lat, lon) pairs as integers in units of 1e-5 degree."""
    q = []
    for la, lo in points_ll:
        v = (int(round(la * SCALE)), int(round(lo * SCALE)))
        if not q or v != q[-1]:
            q.append(v)
    if len(q) < 2:
        return None
    out = [q[0][0], q[0][1]]
    for (a, b), (c, d) in zip(q, q[1:]):
        out += [c - a, d - b]
    return out


# ---------------------------------------------------------------- label placement
def point_in_rings(x, y, rings):
    inside = False
    for ring in rings:
        j = len(ring) - 1
        for i in range(len(ring)):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
            j = i
    return inside


def dist_to_rings(x, y, rings):
    best = float("inf")
    for ring in rings:
        for (ax, ay), (bx, by) in zip(ring, ring[1:] + ring[:1]):
            dx, dy = bx - ax, by - ay
            l2 = dx * dx + dy * dy
            t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / l2))
            best = min(best, math.hypot(x - (ax + t * dx), y - (ay + t * dy)))
    return best


def pole(rings):
    """Approximate pole of inaccessibility: an inside point far from every edge."""
    xs = [p[0] for r in rings for p in r]
    ys = [p[1] for r in rings for p in r]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    best, bd = None, -1.0
    n = 20
    for i in range(n + 1):
        for j in range(n + 1):
            x = x0 + (x1 - x0) * i / n
            y = y0 + (y1 - y0) * j / n
            if point_in_rings(x, y, rings):
                d = dist_to_rings(x, y, rings)
                if d > bd:
                    best, bd = (x, y), d
    if best is None:
        return None
    sx, sy = (x1 - x0) / n, (y1 - y0) / n
    for _ in range(3):
        cx, cy = best
        for i in range(-2, 3):
            for j in range(-2, 3):
                x, y = cx + i * sx / 2, cy + j * sy / 2
                if point_in_rings(x, y, rings):
                    d = dist_to_rings(x, y, rings)
                    if d > bd:
                        best, bd = (x, y), d
        sx, sy = sx / 2, sy / 2
    return best


# ---------------------------------------------------------------- download and processing
def build_query(bbox):
    s, w, n, e = bbox
    bb = "(%.5f,%.5f,%.5f,%.5f)" % (s, w, n, e)
    return (
        '[out:json][timeout:180];('
        'way["highway"~"%s"]%s;'
        'way["natural"="water"]%s;way["waterway"="riverbank"]%s;way["leisure"="park"]%s;'
        'rel["natural"="water"]%s;rel["waterway"="riverbank"]%s;rel["leisure"="park"]%s;'
        ');out geom;' % (ROAD_RE, bb, bb, bb, bb, bb, bb, bb)
    )


def download(query, timeout):
    last = None
    for attempt in range(2):
        for url in ENDPOINTS:
            try:
                print("  asking %s ..." % url)
                req = urllib.request.Request(
                    url,
                    data=urllib.parse.urlencode({"data": query}).encode("utf-8"),
                    headers={"User-Agent": "MSML612-bikeshare-dashboard/1.0 (course project)"},
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                if "elements" in data and data["elements"]:
                    if "remark" in data and "error" in str(data["remark"]).lower():
                        print("  warning: server remark: %s" % data["remark"])
                    return data
                last = "empty response"
            except Exception as ex:  # network, HTTP, JSON: try the next server
                last = ex
                print("    failed: %s" % ex)
        time.sleep(5)
    raise SystemExit(
        "Could not download map data (%s).\nRun with --print-query, open the address in a browser, "
        "save the page as a .json file and use --from-file." % last
    )


def station_bbox(csv_path, pad_lat, pad_lon):
    lats, lons = [], []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lats.append(float(row["lat"]))
            lons.append(float(row["lon"]))
    if not lats:
        raise SystemExit("no stations found in %s" % csv_path)
    return (min(lats) - pad_lat, min(lons) - pad_lon, max(lats) + pad_lat, max(lons) + pad_lon)


def classify_road(tags):
    h = tags.get("highway")
    if h in MAJOR:
        return "major"
    if h in MID:
        return "mid"
    if h in MINOR:
        return "minor"
    return None


def process(data, bbox):
    fr = Frame(bbox)
    sw = fr.to_xy(bbox[0], bbox[1])
    ne = fr.to_xy(bbox[2], bbox[3])
    win = (sw[0], sw[1], ne[0], ne[1])
    roads = {"major": [], "mid": [], "minor": []}
    water, parks, seen = [], [], set()
    cands = []   # (kind, name, area, rings_in_meters)

    def geom_xy(geom):
        return [fr.to_xy(g["lat"], g["lon"]) for g in geom]

    def add_polygon(kind, rings_ll, name):
        meters, enc = [], []
        for ring in rings_ll:
            pts = [fr.to_xy(la, lo) for lo, la in ring]
            pts = clip_polygon(pts, win)
            if len(pts) < 3:
                continue
            pts = rdp(pts + [pts[0]], POLY_TOL)[:-1]
            if len(pts) < 3:
                continue
            meters.append(pts)
            e = encode([fr.to_ll(x, y) for x, y in pts])
            if e:
                enc.append(e)
        if not enc:
            return
        a = max(area(r) for r in meters)
        if a < (MIN_WATER_AREA if kind == "water" else MIN_PARK_AREA):
            return
        key = (kind, tuple(tuple(r[:6]) for r in enc))
        if key in seen:
            return
        seen.add(key)
        (water if kind == "water" else parks).append(enc)
        if name:
            cands.append((kind, name, a, meters))

    for el in data.get("elements", []):
        tags = el.get("tags", {}) or {}
        if el.get("type") == "way" and el.get("geometry"):
            cls = classify_road(tags)
            if cls:
                for run in clip_polyline(geom_xy(el["geometry"]), win):
                    run = rdp(run, ROAD_TOL[cls])
                    e = encode([fr.to_ll(x, y) for x, y in run])
                    if e:
                        roads[cls].append(e)
                continue
            kind = None
            if tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
                kind = "water"
            elif tags.get("leisure") == "park":
                kind = "park"
            geom = el["geometry"]
            if kind and len(geom) >= 4 and geom[0] == geom[-1]:
                ring = [(g["lon"], g["lat"]) for g in geom[:-1]]
                add_polygon(kind, [ring], tags.get("name"))
        elif el.get("type") == "relation":
            kind = None
            if tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
                kind = "water"
            elif tags.get("leisure") == "park":
                kind = "park"
            if not kind:
                continue
            frags = []
            for m in el.get("members", []):
                if m.get("type") == "way" and m.get("geometry") and m.get("role") in ("outer", "inner", ""):
                    frags.append([(g["lon"], g["lat"]) for g in m["geometry"]])
            rings = stitch(frags)
            if rings:
                add_polygon(kind, rings, tags.get("name"))

    labels, used = [], set()
    for kind, name, a, meters in sorted(cands, key=lambda c: -c[2]):
        if name in used or a < MIN_LABEL_AREA[kind] or len(labels) >= MAX_LABELS:
            continue
        p = pole(meters)
        if p is None:
            continue
        la, lo = fr.to_ll(*p)
        labels.append({"t": name, "lat": round(la, 5), "lon": round(lo, 5), "k": kind})
        used.add(name)
    return {
        "v": 1,
        "attribution": "OpenStreetMap contributors",
        "bbox": [round(v, 5) for v in bbox],
        "roads": roads,
        "water": water,
        "parks": parks,
        "labels": labels,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stations", default="data/processed/station_metadata_mapping.csv")
    ap.add_argument("--out", default=str(HERE / "basemap_dc.json"))
    ap.add_argument("--pad-lat", type=float, default=0.045,
                    help="degrees of map beyond the outermost stations (0.045 is about 5 km)")
    ap.add_argument("--pad-lon", type=float, default=0.058,
                    help="if the download times out, retry with --pad-lat 0.03 --pad-lon 0.04")
    ap.add_argument("--from-file", help="use a saved Overpass JSON response instead of downloading")
    ap.add_argument("--save-raw", help="also save the raw download here")
    ap.add_argument("--print-query", action="store_true")
    ap.add_argument("--timeout", type=int, default=240)
    a = ap.parse_args()

    bbox = station_bbox(a.stations, a.pad_lat, a.pad_lon)
    query = build_query(bbox)
    if a.print_query:
        print("Open this address in a browser and save the response as JSON:\n")
        print("https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(query))
        return 0
    if a.from_file:
        data = json.loads(Path(a.from_file).read_text(encoding="utf-8"))
        print("read %s (%d elements)" % (a.from_file, len(data.get("elements", []))))
    else:
        print("Downloading OpenStreetMap data for bbox S%.4f W%.4f N%.4f E%.4f ..." % bbox)
        data = download(query, a.timeout)
        print("received %d elements" % len(data.get("elements", [])))
        if a.save_raw:
            Path(a.save_raw).write_text(json.dumps(data), encoding="utf-8")
    out = process(data, bbox)
    text = json.dumps(out, separators=(",", ":"))
    Path(a.out).write_text(text, encoding="utf-8")
    print(
        "wrote %s (%.0f KB): roads major/mid/minor %d/%d/%d, water %d, parks %d, labels %d"
        % (a.out, len(text) / 1024.0, len(out["roads"]["major"]), len(out["roads"]["mid"]),
           len(out["roads"]["minor"]), len(out["water"]), len(out["parks"]), len(out["labels"]))
    )
    if not any(out["roads"].values()):
        print("WARNING: no roads were written; check the input.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())