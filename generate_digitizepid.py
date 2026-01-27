"""
DigitizePID-native P&ID dataset generator.

Generates synthetic P&ID images with:
- Output/Images/{id}.png
- Output/Annotations/gt_{id}.txt (ICDAR: x1,y1,x2,y2,x3,y3,x4,y4,SymbolName)
- ImagesInfo/{id}/*.npy (symbols, lines, lines2, words, linker, KeyValue, Table)

Uses DigitizePID_Dataset/Classes/ (Pumps, Instruments, Motors, Sensors, Valves) as symbol source.
"""

import cv2
import numpy as np
import random
import math
import string
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
DEFAULT_ROOT = Path(__file__).resolve().parent / "DigitizePID_Dataset"
DEFAULT_WIDTH = 7168
DEFAULT_HEIGHT = 4561
# Match reference samples (7168×4561). Use (3584, 2280) for faster testing.
DEFAULT_CANVAS = (7168, 4561)

BG_GREY = (204, 204, 204)
LINE_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 0, 0)
KEYVALUE_KEYS = [
    "PROJECT/LOCATION/ASSIGN", "ORGANIZATION", "CONTRACTOR PROJECT NO.",
    "CONTRACTOR NO.", "DRAWING NAME", "UNIT", "CONTRACT NAME", "SCALE",
    "DRAW/SHEET NO.", "REV",
]
TABLE_HEADER = ["ISSUE", "DATE", "MADE", "CHECK'D", "APPRV'D", "DESCRIPTION"]

# Reference notes text (28 items) from DigitizePID sample 0 — used when reference dataset is unavailable
REFERENCE_NOTES_28 = [
    "1. PLEASE NOTE EVERY DATA PROVIDED HERE (BELOW COLUMN) IS ENTIRELY RANDOM GENERATED, IT HAS NO RELEVANCE/SIMILARITY WITH ANY OTHER DOCUMENTS OF SIMILAR FIELDS WHATSOEVER.",
    "2. CROSS-CHECKING AT CRITICAL SITES NEAR EFGH VALVES ARE COMPULSORY.",
    "3. TOP CONNECTION.",
    "4. DELETED",
    "5. FOR GENERAL NOTES AND LEGENDS SEE THE DWG-1.23.23.345, DWG-1.23.23.578, DWG-1.23.23.789.",
    "6. ELEVATION ARE TO CENTER LINES UNLESS OTHERWISE NOTED.",
    "7. FOLLOWING XYZ SCHEMES ARE REQUIRED. SEE ABC DWG-1.23.45.67.89 FOR DETAILS.",
    "8. DELETED",
    "9. TEMPERATURE CONTROL IS SET TO ON/OFF TYPE BY DEFAULT.",
    "10. COMPONENTS ARE AT NORMAL PRESSURE WITH POWER RATINGS 15.4 KWh",
    "11. ABC SHUTDOWN IF PQR DETECTS 123 IS NOT WORKING.",
    "12. CLOSED VESSEL HEATER IS SET TO ON/OFF TYPE AT 30°C AND 50°C.",
    "13. DELETED",
    "14. PQR VALVE TO BE INSTALLED AT Q POINT.",
    "15. DELETED",
    "16. ORIENTATION OF ABCD VALVES SHALL BE VERIFIED AND CORRECTED AT SITE IN CASE OF MISMATCH.",
    "17. ALL FIXTURES ARE VALIDATED AND COMPLIANT WITH XYZ.",
    "18. PRIMARY SUPPORT FOR THE EFGH COMPONENTS IS NOT FOUND AT SITE.",
    "19. DELETED",
    "20. DELETED",
    "21. ABC TRANSMITTOR TO BE USED AS BACKUP FOR KMO TRANSMITTOR.",
    "22. BUILDING TO BE IN THE SLANTING PLANE THROUGH THE AXIS OF PQRS TYPE VALVES.",
    "23. DOTTED LINES ARE ALTERNATE.",
    "24. CONNECTION TO EXHAUST IS PROVIDED.",
    "25. ALL DIMENSIONS SHALL BE VERIFIED BY XYZ CONTRACTORS BEFORE INSTALLATIONS/OPERATIONS.",
    "26. DIMENSIONAL TOLERANCE +/- 1MM.",
    "27. DELETED",
    "28. DELETED",
]


def _config(
    root: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_images: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    root = root or DEFAULT_ROOT
    w, h = width or DEFAULT_CANVAS[0], height or DEFAULT_CANVAS[1]
    return {
        "root": Path(root),
        "width": w,
        "height": h,
        "num_images": num_images,
        "seed": seed,
        "classes_dir": Path(root) / "Classes",
        "images_info_dir": Path(root) / "ImagesInfo",
        "output_images_dir": Path(root) / "Output" / "Images",
        "output_annotations_dir": Path(root) / "Output" / "Annotations",
    }


# -----------------------------------------------------------------------------
# Symbol Loader
# -----------------------------------------------------------------------------
class SymbolLoader:
    """Load symbols from DigitizePID_Dataset/Classes/<category>/*.png."""

    def __init__(self, classes_dir: Path, max_symbol_size: int = 100, min_symbol_size: int = 40):
        self.classes_dir = Path(classes_dir)
        self.max_symbol_size = max_symbol_size
        self.min_symbol_size = min_symbol_size
        self.symbols: List[Tuple[np.ndarray, str, str]] = []  # (image, symbol_name, category)
        self._load_all()

    def _load_all(self) -> None:
        if not self.classes_dir.is_dir():
            raise FileNotFoundError(f"Classes directory not found: {self.classes_dir}")
        for cat_dir in sorted(self.classes_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            for p in sorted(cat_dir.glob("*.png")):
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                if img.ndim == 3 and img.shape[2] == 4:
                    alpha = img[:, :, 3]
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    bg = np.ones((*img.shape[:2], 3), dtype=np.uint8) * 255
                    alpha_3d = alpha[:, :, np.newaxis] / 255.0
                    img = (img.astype(np.float64) * alpha_3d + bg.astype(np.float64) * (1 - alpha_3d)).astype(np.uint8)
                elif img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                name = p.stem
                self.symbols.append((img, name, cat_dir.name))

    def get_random_symbol(self) -> Tuple[np.ndarray, str, str]:
        """Returns (image, symbol_name, category). Image is resized for placement."""
        img, name, cat = random.choice(self.symbols)
        h, w = img.shape[:2]
        s = random.randint(self.min_symbol_size, self.max_symbol_size) / max(h, w)
        nw, nh = max(10, int(w * s)), max(10, int(h * s))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return img, name, cat

    def get_random_symbols(self, n: int) -> List[Tuple[np.ndarray, str, str]]:
        out = []
        for _ in range(n):
            out.append(self.get_random_symbol())
        return out


# -----------------------------------------------------------------------------
# Canvas regions (main diagram, notes, title block)
# -----------------------------------------------------------------------------
def canvas_regions(
    width: int,
    height: int,
    *,
    outer_margin: Optional[int] = None,
    notes_height: Optional[int] = None,
    title_height: Optional[int] = None,
    right_col_pad: Optional[int] = None,
    blocks_border_margin: Optional[int] = None,
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Returns (x1,y1,x2,y2) for main_diagram, notes, title_block.
    Main stays inside image; notes top-right, title bottom-right.
    blocks_border_margin = extra margin between outer border and notes/title blocks (right, top, bottom).
    """
    ref_h = 4561
    ref_w = 7168
    scale_h = height / ref_h
    scale_w = width / ref_w
    scale = max(scale_w, scale_h)

    # Outer margin so main block never touches/crosses image border
    M = outer_margin if outer_margin is not None else max(80, int(90 * scale))
    # Extra margin for notes/title blocks from border (right edge, top of notes, bottom of title)
    B = blocks_border_margin if blocks_border_margin is not None else max(60, int(70 * scale))
    # Padding between main diagram and right column (notes/title)
    col_pad = right_col_pad if right_col_pad is not None else max(50, int(60 * scale))
    # Right column: main ends, then gap, then notes/title (inset by B from right border)
    main_right = int(0.68 * width) - col_pad
    right_x1 = main_right + col_pad
    right_x2 = width - M - B  # more margin from right border

    # Notes block: top-right, with extra top margin B from border
    notes_h = notes_height if notes_height is not None else max(680, int(750 * scale_h))
    notes = (right_x1, M + B, right_x2, M + B + notes_h)

    # Title block: bottom-right, with extra bottom margin B from border (height increased to avoid overflow)
    title_h = title_height if title_height is not None else max(680, int(760 * scale_h))
    title_block = (right_x1, height - M - B - title_h, right_x2, height - M - B)

    # Main diagram: left of right column, fully inside outer margin (never crosses border)
    main_diagram = (M, M, main_right, height - M)

    return {"main_diagram": main_diagram, "notes": notes, "title_block": title_block}


# -----------------------------------------------------------------------------
# Load reference KeyValue, Table, Notes from DigitizePID_Dataset/{0..14}/
# -----------------------------------------------------------------------------
def load_reference_content(root: Path, ref_sample_id: int, ref_width: int = 7168) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    """Load KeyValue, Table, and notes list from reference sample. Returns (keyvalue, table, notes_list)."""
    root = Path(root)
    sid = ref_sample_id % 15
    kv_path = root / str(sid) / f"{sid}_KeyValue.npy"
    tb_path = root / str(sid) / f"{sid}_Table.npy"
    wd_path = root / str(sid) / f"{sid}_words.npy"
    keyvalue, table = None, None
    notes_list = list(REFERENCE_NOTES_28)

    if kv_path.exists():
        try:
            keyvalue = np.load(str(kv_path), allow_pickle=True)
        except Exception:
            pass
    if tb_path.exists():
        try:
            table = np.load(str(tb_path), allow_pickle=True)
        except Exception:
            pass
    if wd_path.exists():
        try:
            wd = np.load(str(wd_path), allow_pickle=True)
            # Notes: words with x_center > 0.7*ref_width, sort by y, group by "N." into lines
            right = []
            for w in wd:
                bbox = w[1]
                if hasattr(bbox, "__len__") and len(bbox) >= 4:
                    xc = (bbox[0] + bbox[2]) / 2
                    if xc > 0.7 * ref_width:
                        right.append((float(bbox[1]), str(w[2])[:200]))
            right.sort(key=lambda x: x[0])
            cur, notes_list = [], []
            for y, t in right:
                t = t.strip()
                if t == "NOTES":
                    continue
                if len(t) <= 4 and t.rstrip(".").isdigit():
                    if cur:
                        notes_list.append(" ".join(cur))
                    cur = [t]
                else:
                    cur.append(t)
            if cur:
                notes_list.append(" ".join(cur))
            if len(notes_list) >= 20:
                notes_list = notes_list[:28]
        except Exception:
            pass
    return keyvalue, table, notes_list


# -----------------------------------------------------------------------------
# Placement: symbols on grid (aligned, angle 0) for reference-like layout
# -----------------------------------------------------------------------------
def _get_bbox_rect(x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = x - w // 2
    y1 = y - h // 2
    return (x1, y1, x1 + w, y1 + h)


def _rects_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], pad: int = 20) -> bool:
    ax1, ay1, ax2, ay2 = a[0] - pad, a[1] - pad, a[2] + pad, a[3] + pad
    bx1, by1, bx2, by2 = b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)


def _get_polygon_tl_tr_br_bl(x: int, y: int, w: int, h: int, angle_deg: float = 0) -> List[Tuple[int, int]]:
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    if angle_deg != 0:
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        R = np.array([[c, -s], [s, c]])
        corners = corners @ R.T
    corners[:, 0] += x
    corners[:, 1] += y
    pts = [(int(c[0]), int(c[1])) for c in corners]
    by_y = sorted(pts, key=lambda p: (p[1], p[0]))
    tl = min(by_y[:2], key=lambda p: p[0])
    tr = max(by_y[:2], key=lambda p: p[0])
    bl = min(by_y[2:], key=lambda p: p[0])
    br = max(by_y[2:], key=lambda p: p[0])
    return [tl, tr, br, bl]


def _get_mid_edges(x: int, y: int, w: int, h: int, angle_deg: float) -> List[Tuple[int, int]]:
    hw, hh = w / 2, h / 2
    pts = [(0, -hh), (hw, 0), (0, hh), (-hw, 0)]
    if angle_deg != 0:
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        R = np.array([[c, -s], [s, c]])
        pts = [((p[0] * c - p[1] * s), (p[0] * s + p[1] * c)) for p in pts]
    return [(int(x + p[0]), int(y + p[1])) for p in pts]


def _grid_cells(main_rect: Tuple[int, int, int, int], step_x: int = 100, step_y: int = 90,
                jitter: int = 8) -> List[Tuple[int, int]]:
    """Return list of (cx, cy) grid centers inside main_rect with optional jitter."""
    x1, y1, x2, y2 = main_rect
    cells = []
    y = y1 + step_y // 2
    while y < y2 - step_y // 2:
        x = x1 + step_x // 2
        while x < x2 - step_x // 2:
            jx = x + random.randint(-jitter, jitter) if jitter else x
            jy = y + random.randint(-jitter, jitter) if jitter else y
            jx = max(x1 + step_x // 4, min(x2 - step_x // 4, jx))
            jy = max(y1 + step_y // 4, min(y2 - step_y // 4, jy))
            cells.append((jx, jy))
            x += step_x
        y += step_y
    random.shuffle(cells)
    return cells


def place_symbols(
    symbols: List[Tuple[np.ndarray, str, str]],
    main_rect: Tuple[int, int, int, int],
    max_attempts_per_symbol: int = 120,
    use_grid: bool = True,
    grid_step_x: int = 100,
    grid_step_y: int = 90,
) -> List[Dict[str, Any]]:
    """Place symbols in main_rect. use_grid=True gives aligned grid layout (reference-like)."""
    x1, y1, x2, y2 = main_rect
    placed: List[Dict[str, Any]] = []

    if use_grid:
        # jitter=0 for strict horizontal/vertical alignment (reference-like)
        cells = _grid_cells(main_rect, step_x=grid_step_x, step_y=grid_step_y, jitter=0)
        for idx, (img, name, cat) in enumerate(symbols):
            if idx >= len(cells):
                break
            cx, cy = cells[idx]
            h, w = img.shape[:2]
            angle = 0.0  # aligned, no rotation
            rw, rh = w, h
            rect = _get_bbox_rect(cx, cy, rw, rh)
            if rect[0] < x1 or rect[1] < y1 or rect[2] > x2 or rect[3] > y2:
                continue
            if any(_rects_overlap(rect, _get_bbox_rect(p["x"], p["y"], p["rotated_w"], p["rotated_h"])) for p in placed):
                continue
            poly = _get_polygon_tl_tr_br_bl(cx, cy, w, h, angle)
            edges = _get_mid_edges(cx, cy, w, h, angle)
            placed.append({
                "symbol_id": f"symbol_{idx + 1}",
                "x": cx, "y": cy,
                "image": img,
                "symbol_name": name,
                "category": cat,
                "angle": angle,
                "w": w, "h": h,
                "rotated_w": rw, "rotated_h": rh,
                "bbox": (cx - rw // 2, cy - rh // 2, cx + rw // 2, cy + rh // 2),
                "polygon": poly,
                "mid_edges": edges,
            })
        return placed

    # Fallback: random placement with collision check
    for idx, (img, name, cat) in enumerate(symbols):
        h, w = img.shape[:2]
        angle = random.uniform(-8, 8)
        rad = math.radians(abs(angle))
        rw = int(h * abs(math.sin(rad)) + w * abs(math.cos(rad))) + 4
        rh = int(h * abs(math.cos(rad)) + w * abs(math.sin(rad))) + 4
        rw, rh = max(rw, w), max(rh, h)
        for _ in range(max_attempts_per_symbol):
            cx = random.randint(x1 + rw // 2 + 10, x2 - rw // 2 - 10)
            cy = random.randint(y1 + rh // 2 + 10, y2 - rh // 2 - 10)
            rect = _get_bbox_rect(cx, cy, rw, rh)
            if rect[0] < x1 or rect[1] < y1 or rect[2] > x2 or rect[3] > y2:
                continue
            if any(_rects_overlap(rect, _get_bbox_rect(p["x"], p["y"], p["rotated_w"], p["rotated_h"])) for p in placed):
                continue
            poly = _get_polygon_tl_tr_br_bl(cx, cy, w, h, angle)
            edges = _get_mid_edges(cx, cy, w, h, angle)
            placed.append({
                "symbol_id": f"symbol_{idx + 1}",
                "x": cx, "y": cy,
                "image": img,
                "symbol_name": name,
                "category": cat,
                "angle": angle,
                "w": w, "h": h,
                "rotated_w": rw, "rotated_h": rh,
                "bbox": (cx - rw // 2, cy - rh // 2, cx + rw // 2, cy + rh // 2),
                "polygon": poly,
                "mid_edges": edges,
            })
            break
    return placed


# -----------------------------------------------------------------------------
# Line routing (Manhattan 3-segment), path-cross check
# -----------------------------------------------------------------------------
def _segments_intersect(a1: Tuple[int, int], a2: Tuple[int, int],
                        b1: Tuple[int, int], b2: Tuple[int, int]) -> bool:
    def o(p, q, r):
        v = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        return 0 if v == 0 else (1 if v > 0 else 2)

    def on(p, q, r):
        return (min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and
                min(p[1], r[1]) <= q[1] <= max(p[1], r[1]))

    o1, o2 = o(a1, a2, b1), o(a1, a2, b2)
    o3, o4 = o(b1, b2, a1), o(b1, b2, a2)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on(a1, b1, a2): return True
    if o2 == 0 and on(a1, b2, a2): return True
    if o3 == 0 and on(b1, a1, b2): return True
    if o4 == 0 and on(b1, a2, b2): return True
    return False


def _polygon_edges(poly: List[Tuple[int, int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def path_crosses_symbol(
    pt1: Tuple[int, int], pt2: Tuple[int, int],
    placed: List[Dict], exclude: Tuple[int, int],
) -> bool:
    for i, p in enumerate(placed):
        if i == exclude[0] or i == exclude[1]:
            continue
        for e1, e2 in _polygon_edges(p["polygon"]):
            if _segments_intersect(pt1, pt2, e1, e2):
                return True
    return False


def _manhattan_segments(e1: Tuple[int, int], e2: Tuple[int, int], use_vhv: bool
                       ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    x1, y1 = e1
    x2, y2 = e2
    if use_vhv:
        m1 = (x1, (y1 + y2) // 2)
        m2 = (x2, (y1 + y2) // 2)
        return [(e1, m1), (m1, m2), (m2, e2)]
    else:
        m1 = ((x1 + x2) // 2, y1)
        m2 = ((x1 + x2) // 2, y2)
        return [(e1, m1), (m1, m2), (m2, e2)]


def path_crosses_symbol_3(e1: Tuple[int, int], e2: Tuple[int, int],
                         placed: List[Dict], exclude: Tuple[int, int], use_vhv: bool) -> bool:
    for seg in _manhattan_segments(e1, e2, use_vhv):
        if path_crosses_symbol(seg[0], seg[1], placed, exclude):
            return True
    return False


def _gen_line_spec() -> str:
    inch = random.choice(["2", "3", "4", "5", "6", "7", "8", "10", "12", "14"])
    code = "".join(random.choices(string.ascii_uppercase, k=2))
    num = random.randint(1000, 9999)
    return f'{inch}"-{code}-{num}'


def build_connections(
    placed: List[Dict],
    main_rect: Tuple[int, int, int, int],
    connection_prob: float = 0.65,
    distance_threshold: int = 320,
    max_connections_per_symbol: int = 4,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (lines, linker_updates). Phase 1: ensure every symbol has ≥1 connection. Phase 2: add more.
    """
    lines: List[Dict] = []
    linker_add: List[Tuple[int, str]] = []
    conn_count = {i: 0 for i in range(len(placed))}
    made: set = set()  # (min(i,j), max(i,j)) already connected

    def try_add(i: int, j: int, e1: Tuple[int, int], e2: Tuple[int, int], use_vhv: bool) -> bool:
        if conn_count[i] >= max_connections_per_symbol or conn_count[j] >= max_connections_per_symbol:
            return False
        pair = (min(i, j), max(i, j))
        if pair in made:
            return False
        line_id = f"line_{len(lines) + 1}"
        spec = _gen_line_spec()
        style = random.choice(["solid", "dashed"])
        # Always use Manhattan (horizontal/vertical only), never diagonal
        segs = _manhattan_segments(e1, e2, use_vhv)
        seg_coords = [(s[0][0], s[0][1], s[1][0], s[1][1]) for s in segs]
        lines.append({"line_id": line_id, "segments": seg_coords, "spec": spec, "style": style})
        linker_add.append((i, line_id))
        linker_add.append((j, line_id))
        conn_count[i] += 1
        conn_count[j] += 1
        made.add(pair)
        return True

    # Build potential connections (i, j, dist, e1, e2, use_3)
    potential = []
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            pi, pj = placed[i], placed[j]
            dist = math.hypot(pi["x"] - pj["x"], pi["y"] - pj["y"])
            if dist > distance_threshold:
                continue
            edgs_i, edgs_j = pi["mid_edges"], pj["mid_edges"]
            best_d, best_e1, best_e2, best_vhv = float("inf"), None, None, None
            # Only consider Manhattan paths (horizontal/vertical segments), never diagonal
            for e1 in edgs_i:
                for e2 in edgs_j:
                    d = math.hypot(e1[0] - e2[0], e1[1] - e2[1])
                    for use_vhv in [True, False]:
                        if not path_crosses_symbol_3(e1, e2, placed, (i, j), use_vhv):
                            if d < best_d:
                                best_d, best_e1, best_e2, best_vhv = d, e1, e2, use_vhv
            if best_e1 is not None and best_e2 is not None and best_vhv is not None:
                potential.append((i, j, dist, best_e1, best_e2, best_vhv))

    potential.sort(key=lambda t: t[2])  # nearest first

    # Phase 1: ensure every symbol has at least one connection
    for i, j, _, e1, e2, use_vhv in potential:
        if conn_count[i] == 0 or conn_count[j] == 0:
            try_add(i, j, e1, e2, use_vhv)

    # Phase 2: add more connections
    for i, j, _, e1, e2, use_vhv in potential:
        if random.random() <= connection_prob:
            try_add(i, j, e1, e2, use_vhv)

    return lines, linker_add


# -----------------------------------------------------------------------------
# Text: tags per symbol, line specs
# -----------------------------------------------------------------------------
def _gen_tag() -> str:
    pref = "".join(random.choices(string.ascii_uppercase, k=2))
    num = random.randint(10000, 99999)
    return f"{pref}-{num}"


def build_words_and_linker(
    placed: List[Dict],
    lines: List[Dict],
    linker_line: List[Tuple[int, str]],
    font_scale: float = 0.5,
    thickness: int = 1,
) -> Tuple[List[Dict], Dict[int, List[str]]]:
    """
    For each symbol, assign a tag and bbox (approx). For each line, place spec text and bbox.
    Returns (words_list, symbol_idx -> [word_id, line_id, ...] for linker).
    """
    words: List[Dict] = []
    symbol_refs: Dict[int, List[str]] = {i: [] for i in range(len(placed))}
    # Font for bbox estimation
    font = cv2.FONT_HERSHEY_SIMPLEX
    base_h = 20
    for i, p in enumerate(placed):
        tag = _gen_tag()
        cx, cy = p["x"], p["y"]
        bx1 = cx - 60
        by1 = cy - p["rotated_h"] // 2 - base_h - 5
        bx2 = bx1 + 80
        by2 = by1 + base_h
        wid = f"word_{len(words) + 1}"
        words.append({"word_id": wid, "bbox": [bx1, by1, bx2, by2], "text": tag, "flags": 0})
        symbol_refs[i].append(wid)
    for (line_idx_ish, line_id) in linker_line:
        # line_idx_ish is symbol index; we attach line_id to that symbol
        symbol_refs[line_idx_ish].append(line_id)
    # Line spec words: one per line, placed at first segment mid
    for line in lines:
        seg = line["segments"][0]
        mx = (seg[0] + seg[2]) // 2
        my = (seg[1] + seg[3]) // 2
        bx1, by1 = mx - 50, my - 12
        bx2, by2 = mx + 50, my + 12
        wid = f"word_{len(words) + 1}"
        words.append({"word_id": wid, "bbox": [bx1, by1, bx2, by2], "text": line["spec"], "flags": 0})
    return words, symbol_refs


# -----------------------------------------------------------------------------
# KeyValue and Table templates
# -----------------------------------------------------------------------------
def make_keyvalue(sample_id: int, width: int, height: int) -> np.ndarray:
    arr = np.array([
        [KEYVALUE_KEYS[0], f"{''.join(random.choices(string.ascii_uppercase, k=2))}-{random.randint(10,99)}-{random.randint(10,99)}"],
        [KEYVALUE_KEYS[1], f"P-{random.randint(0,9)}"],
        [KEYVALUE_KEYS[2], f"{''.join(random.choices(string.ascii_uppercase, k=2))}-{random.randint(1000,9999)}"],
        [KEYVALUE_KEYS[3], str(random.randint(100, 999))],
        [KEYVALUE_KEYS[4], f"SAMPLE_{random.randint(1000,9999)}.PNG"],
        [KEYVALUE_KEYS[5], f"{random.randint(100,999)}-{random.randint(1000,9999)}"],
        [KEYVALUE_KEYS[6], "PROJ. XYZ P&ID"],
        [KEYVALUE_KEYS[7], random.choice(["None", "Std.", "NTS"])],
        [KEYVALUE_KEYS[8], str(random.randint(80000000, 99999999))],
        [KEYVALUE_KEYS[9], str(random.randint(1, 5))],
    ], dtype=object)
    return arr


def make_table() -> np.ndarray:
    rows = [TABLE_HEADER]
    for i, letter in enumerate(["A", "B", "C", "D"]):
        rows.append([
            letter,
            f"{random.randint(1,28):02d}/{random.randint(1,12):02d}/{random.randint(0,9)}2" if random.random() > 0.5 else f"{random.randint(1,28)}{random.choice('JanFebMarAprMayJunJulAugSepOctNovDec')}{random.randint(0,9)}",
            "".join(random.choices(string.ascii_uppercase, k=random.randint(2,4))),
            "".join(random.choices(string.ascii_uppercase, k=random.randint(2,3))),
            "".join(random.choices(string.ascii_uppercase, k=random.randint(2,4))),
            f"ISSUE CONSTR. REV.{i+1}",
        ])
    return np.array(rows, dtype=object)


# -----------------------------------------------------------------------------
# Renderer: draw canvas, lines, symbols, text, notes, title block
# -----------------------------------------------------------------------------
def draw_dashed_line(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                     color: Tuple[int, int, int], thickness: int, dash_len: int = 12) -> None:
    x1, y1 = pt1
    x2, y2 = pt2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1:
        return
    n = max(1, int(length / dash_len))
    for k in range(n):
        t0 = k / n
        t1 = (k + 0.5) / n
        if t1 > 1:
            t1 = 1
        px0 = int(x1 + t0 * (x2 - x1))
        py0 = int(y1 + t0 * (y2 - y1))
        px1 = int(x1 + t1 * (x2 - x1))
        py1 = int(y1 + t1 * (y2 - y1))
        cv2.line(img, (px0, py0), (px1, py1), color, thickness)


def draw_line_segment(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                      color: Tuple[int, int, int], thickness: int, dashed: bool) -> None:
    if dashed:
        draw_dashed_line(img, (x1, y1), (x2, y2), color, thickness)
    else:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_symbol(img: np.ndarray, sym: Dict, bg_color: Tuple[int, int, int]) -> None:
    cx, cy = sym["x"], sym["y"]
    sym_img = sym["image"]
    angle = sym["angle"]
    h, w = sym_img.shape[:2]
    if angle != 0:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        sym_img = cv2.warpAffine(sym_img, M, (w, h), borderValue=bg_color, flags=cv2.INTER_LINEAR)
    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(img.shape[1], cx + w // 2)
    y2 = min(img.shape[0], cy + h // 2)
    sx1 = max(0, w // 2 - (cx - x1))
    sy1 = max(0, h // 2 - (cy - y1))
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    if sx2 <= sx1 or sy2 <= sy1:
        return
    patch = sym_img[sy1:sy2, sx1:sx2]
    if patch.size == 0:
        return
    ph, pw = patch.shape[:2]
    roi = img[y1:y1 + ph, x1:x1 + pw]
    if roi.shape[0] != ph or roi.shape[1] != pw:
        return
    # P&ID look: symbol non-white pixels -> black on background
    if patch.ndim == 3:
        mask = np.any(patch < 240, axis=2)
    else:
        mask = patch < 240
    if mask.shape != roi.shape[:2]:
        return
    roi[mask] = 0


def render_canvas(
    width: int,
    height: int,
    placed: List[Dict],
    lines: List[Dict],
    words: List[Dict],
    keyvalue: np.ndarray,
    table: np.ndarray,
    regions: Dict[str, Tuple[int, int, int, int]],
    notes_list: Optional[List[str]] = None,
    sheet_margin: int = 40,
) -> np.ndarray:
    """Draw P&ID. All three regions get dashed borders; outer sheet has dashed border. notes_list from reference."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = BG_GREY
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = width / 3584.0
    fs = max(0.35, min(0.7, 0.45 * scale))
    th = max(1, int(round(scale)))
    line_w = max(1, min(3, int(round(scale))))
    border_lw = max(1, int(2 * scale))
    margin = sheet_margin  # same as canvas_regions outer_margin so main never crosses border

    # 1) Draw piping lines
    for line in lines:
        lw = line_w
        dashed = line["style"] == "dashed"
        for (xa, ya, xb, yb) in line["segments"]:
            draw_line_segment(img, xa, ya, xb, yb, LINE_COLOR, lw, dashed)

    # 2) Draw symbols
    for sym in placed:
        draw_symbol(img, sym, BG_GREY)

    # 3) Draw text (tags and line specs)
    for w in words:
        bbox = w["bbox"]
        x, y = bbox[0], bbox[3] - max(2, int(4 * scale))
        if x < 0 or y < 0 or y > height:
            continue
        txt = w["text"]
        cv2.putText(img, txt, (x, y), font, fs, TEXT_COLOR, th, cv2.LINE_AA)

    # 4) Outer dashed border (entire sheet)
    ox1, oy1 = margin, margin
    ox2, oy2 = width - margin, height - margin
    draw_dashed_line(img, (ox1, oy1), (ox2, oy1), LINE_COLOR, border_lw)
    draw_dashed_line(img, (ox2, oy1), (ox2, oy2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (ox2, oy2), (ox1, oy2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (ox1, oy2), (ox1, oy1), LINE_COLOR, border_lw)

    # 5) Dashed border around main diagram
    mx1, my1, mx2, my2 = regions["main_diagram"]
    draw_dashed_line(img, (mx1, my1), (mx2, my1), LINE_COLOR, border_lw)
    draw_dashed_line(img, (mx2, my1), (mx2, my2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (mx2, my2), (mx1, my2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (mx1, my2), (mx1, my1), LINE_COLOR, border_lw)

    # 6) Notes block (top-right) — dashed border + content with inner padding
    nx1, ny1, nx2, ny2 = regions["notes"]
    draw_dashed_line(img, (nx1, ny1), (nx2, ny1), LINE_COLOR, border_lw)
    draw_dashed_line(img, (nx2, ny1), (nx2, ny2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (nx2, ny2), (nx1, ny2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (nx1, ny2), (nx1, ny1), LINE_COLOR, border_lw)
    note_pad = max(12, int(18 * scale))  # inner padding so content doesn't touch border
    note_fs = max(0.28, 0.32 * scale)
    note_dy = int(26 * scale)
    cv2.putText(img, "NOTES", (nx1 + note_pad, ny1 + note_dy), font, max(0.4, 0.5 * scale), TEXT_COLOR, th, cv2.LINE_AA)
    nlist = notes_list or REFERENCE_NOTES_28
    line_h = int(20 * scale)
    for i, text in enumerate(nlist[:28]):
        y_pos = ny1 + note_dy + (i + 1) * line_h
        if y_pos >= ny2 - note_pad:
            break
        txt = text[:75] + ".." if len(text) > 75 else text
        cv2.putText(img, txt, (nx1 + note_pad, y_pos), font, note_fs, TEXT_COLOR, th, cv2.LINE_AA)

    # 7) Title block (bottom-right) — dashed border + KeyValue/Table with inner padding
    tx1, ty1, tx2, ty2 = regions["title_block"]
    draw_dashed_line(img, (tx1, ty1), (tx2, ty1), LINE_COLOR, border_lw)
    draw_dashed_line(img, (tx2, ty1), (tx2, ty2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (tx2, ty2), (tx1, ty2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (tx1, ty2), (tx1, ty1), LINE_COLOR, border_lw)
    title_pad = max(12, int(18 * scale))  # inner padding so content doesn't touch border
    y_max = ty2 - title_pad  # do not draw below this
    y_off = ty1 + int(22 * scale)
    kv_fs = max(0.28, 0.32 * scale)
    kv_line_h = int(20 * scale)
    for row in keyvalue:
        if y_off <= y_max:
            cv2.putText(img, f"{row[0]}: {row[1]}", (tx1 + title_pad, y_off), font, kv_fs, TEXT_COLOR, th, cv2.LINE_AA)
        y_off += kv_line_h
    y_off += int(8 * scale)
    tbl_line_h = int(18 * scale)
    for r in range(table.shape[0]):
        if y_off <= y_max:
            x_off = tx1 + title_pad
            for c in range(table.shape[1]):
                cv2.putText(img, str(table[r, c])[:14], (x_off, y_off), font, max(0.22, 0.26 * scale), TEXT_COLOR, th, cv2.LINE_AA)
                x_off += int(85 * scale)
        y_off += tbl_line_h

    return img


# -----------------------------------------------------------------------------
# Writers: ImagesInfo/*.npy and Output/Annotations/gt_{id}.txt
# -----------------------------------------------------------------------------
def build_npy_arrays(
    sample_id: int,
    placed: List[Dict],
    lines: List[Dict],
    words: List[Dict],
    symbol_refs: Dict[int, List[str]],
    keyvalue: np.ndarray,
    table: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build arrays in the same schema as reference DigitizePID .npy files."""
    pid = str(sample_id)
    # symbols: (N, 3) object [symbol_id, bbox [x1,y1,x2,y2], class_id]
    # class_id: use symbol index 1..N or category hash; use string "1".."32" - we use (index % 32)+1 for compatibility
    symbols_arr = []
    for i, p in enumerate(placed):
        bbox = list(p["bbox"])
        class_id = str((i % 32) + 1)
        symbols_arr.append([p["symbol_id"], bbox, class_id])
    symbols_npy = np.array(symbols_arr, dtype=object)

    # words: (M, 4) object [word_id, bbox, text, flags]
    words_arr = [[w["word_id"], w["bbox"], w["text"], w["flags"]] for w in words]
    words_npy = np.array(words_arr, dtype=object)

    # lines: (L, 4) object [line_id, [x1,y1,x2,y2], spec, style] — one row per logical line, first segment used for coords
    lines_arr = []
    for line in lines:
        seg = line["segments"][0]
        lines_arr.append([line["line_id"], [int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])], line["spec"], line["style"]])
    lines_npy = np.array(lines_arr, dtype=object)

    # lines2: (L2, 5) int64 — all segments flattened
    lines2_list = []
    for line in lines:
        for seg in line["segments"]:
            typ = 1 if line["style"] == "solid" else 0
            lines2_list.append([seg[0], seg[1], seg[2], seg[3], typ])
    lines2_npy = np.array(lines2_list, dtype=np.int64) if lines2_list else np.zeros((0, 5), dtype=np.int64)

    # linker: (N, 2) object [symbol_id, list of "word_*" and "line_*"]
    linker_arr = []
    for i, p in enumerate(placed):
        ids = symbol_refs.get(i, [])
        linker_arr.append([p["symbol_id"], list(ids)])
    linker_npy = np.array(linker_arr, dtype=object)

    return {
        f"{pid}_symbols.npy": symbols_npy,
        f"{pid}_words.npy": words_npy,
        f"{pid}_lines.npy": lines_npy,
        f"{pid}_lines2.npy": lines2_npy,
        f"{pid}_linker.npy": linker_npy,
        f"{pid}_KeyValue.npy": keyvalue,
        f"{pid}_Table.npy": table,
    }


def write_images_info(sample_id: int, out_dir: Path, data: Dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    id_dir = out_dir / str(sample_id)
    id_dir.mkdir(parents=True, exist_ok=True)
    for fname, arr in data.items():
        np.save(id_dir / fname, arr, allow_pickle=True)


def write_gt_txt(placed: List[Dict], path: Path, width: int, height: int) -> None:
    """ICDAR format: one line per symbol = x1,y1,x2,y2,x3,y3,x4,y4,SymbolName"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in placed:
            poly = p["polygon"]
            # tl, tr, br, bl -> x1,y1,x2,y2,x3,y3,x4,y4
            coords = []
            for pt in poly:
                x = max(0, min(width - 1, pt[0]))
                y = max(0, min(height - 1, pt[1]))
                coords.extend([x, y])
            line = ",".join(map(str, coords)) + "," + p["symbol_name"] + "\n"
            f.write(line)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def generate_one(
    sample_id: int,
    loader: SymbolLoader,
    width: int,
    height: int,
    cfg: Dict[str, Any],
) -> None:
    sheet_margin = max(80, int(90 * width / 3584))
    blocks_border_margin = max(60, int(70 * width / 3584))  # extra margin for notes/title from border
    regions = canvas_regions(
        width,
        height,
        outer_margin=sheet_margin,
        right_col_pad=max(50, int(60 * width / 3584)),
        blocks_border_margin=blocks_border_margin,
    )
    main_rect = regions["main_diagram"]

    # Load reference KeyValue, Table, Notes from DigitizePID_Dataset/{0..14}/
    ref_kv, ref_tb, notes_list = load_reference_content(cfg["root"], sample_id, ref_width=width)
    keyvalue = ref_kv if ref_kv is not None else make_keyvalue(sample_id, width, height)
    table = ref_tb if ref_tb is not None else make_table()

    # Grid placement for aligned, reference-like layout
    step = max(80, int(100 * width / 3584))
    cell_count = len(_grid_cells(main_rect, step_x=step, step_y=int(step * 0.9), jitter=0))
    num_sym = min(cell_count, max(30, random.randint(50, 90)))
    symbols = loader.get_random_symbols(num_sym)
    placed = place_symbols(
        symbols,
        main_rect,
        use_grid=True,
        grid_step_x=step,
        grid_step_y=int(step * 0.9),
    )
    if len(placed) == 0:
        print(f"  [sample {sample_id}] no symbols placed, skip")
        return

    # More connections: ensure every symbol has ≥1, then add more (reference has ~2x lines vs symbols)
    lines, linker_line = build_connections(
        placed,
        main_rect,
        connection_prob=0.65,
        distance_threshold=max(280, int(320 * width / 3584)),
        max_connections_per_symbol=4,
    )
    words, symbol_refs = build_words_and_linker(placed, lines, linker_line)

    canvas = render_canvas(
        width,
        height,
        placed,
        lines,
        words,
        keyvalue,
        table,
        regions,
        notes_list=notes_list,
        sheet_margin=sheet_margin,
    )

    # Save image
    out_img_dir = cfg["output_images_dir"]
    out_img_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_img_dir / f"{sample_id}.png"
    cv2.imwrite(str(img_path), canvas)

    # Save annotations (gt)
    gt_path = cfg["output_annotations_dir"] / f"gt_{sample_id}.txt"
    write_gt_txt(placed, gt_path, width, height)

    # Save ImagesInfo/*.npy
    npy_data = build_npy_arrays(sample_id, placed, lines, words, symbol_refs, keyvalue, table)
    write_images_info(sample_id, cfg["images_info_dir"], npy_data)


def run(
    root: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_images: int = 10,
    seed: Optional[int] = None,
) -> None:
    cfg = _config(root=root, width=width, height=height, num_images=num_images, seed=seed)
    if cfg["seed"] is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    # Larger symbols at full resolution (7168×4561) to match reference look
    w = cfg["width"]
    sym_max = 120 if w >= 5000 else 90
    sym_min = 50 if w >= 5000 else 36
    loader = SymbolLoader(cfg["classes_dir"], max_symbol_size=sym_max, min_symbol_size=sym_min)
    if not loader.symbols:
        raise FileNotFoundError(f"No symbols loaded from {cfg['classes_dir']}")

    w, h = cfg["width"], cfg["height"]
    print(f"DigitizePID generator: {num_images} images, canvas {w}x{h}, symbols={len(loader.symbols)}")
    for i in range(num_images):
        generate_one(i, loader, w, h, cfg)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  generated {i + 1}/{num_images}")
    print(f"Done. Images: {cfg['output_images_dir']}, Annotations: {cfg['output_annotations_dir']}, ImagesInfo: {cfg['images_info_dir']}")


if __name__ == "__main__":
    run(
        root=DEFAULT_ROOT,
        width=DEFAULT_CANVAS[0],
        height=DEFAULT_CANVAS[1],
        num_images=10,
        seed=42,
    )
