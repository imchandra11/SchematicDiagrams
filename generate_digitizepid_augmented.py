"""
DigitizePID augmented P&ID dataset generator.

Reuses logic from generate_digitizepid.py and adds:
- Pixel-level augmentation (blur, noise, environmental effects) from generate_enhanced_dataset.py
- Yellowish paper effect
- NOTES header bar and TITLE BLOCK header bar (like reference images)
- Larger font sizes for notes and title-block text

Output only:
- DigitizePID_Dataset/Output/Images/0.png, 1.png, ...
- DigitizePID_Dataset/Output/Annotations/gt_0.txt, gt_1.txt, ...

Does NOT save ImagesInfo or any .npy files.
Uses DigitizePID_Dataset/Classes/ as symbol source.
"""

import cv2
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Reuse core logic from generate_digitizepid
from generate_digitizepid import (
    canvas_regions,
    load_reference_content,
    place_symbols,
    build_connections,
    build_words_and_linker,
    make_keyvalue,
    make_table,
    draw_dashed_line,
    draw_line_segment,
    draw_symbol,
    write_gt_txt,
    REFERENCE_NOTES_28,
    SymbolLoader,
    _grid_cells,
    BG_GREY,
    LINE_COLOR,
    TEXT_COLOR,
)

from generate_enhanced_dataset import PixelLevelTransformProcessor

# -----------------------------------------------------------------------------
# Config (no ImagesInfo)
# -----------------------------------------------------------------------------
DEFAULT_ROOT = Path(__file__).resolve().parent / "DigitizePID_Dataset"
DEFAULT_CANVAS = (7168, 4561)


def _config(
    root: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_images: int = 10,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    root = Path(root or DEFAULT_ROOT)
    w = width or DEFAULT_CANVAS[0]
    h = height or DEFAULT_CANVAS[1]
    return {
        "root": root,
        "width": w,
        "height": h,
        "num_images": num_images,
        "seed": seed,
        "classes_dir": root / "Classes",
        "output_images_dir": root / "Output" / "Images",
        "output_annotations_dir": root / "Output" / "Annotations",
    }


# -----------------------------------------------------------------------------
# Yellowish paper effect (RGB image)
# -----------------------------------------------------------------------------
def apply_yellowish_paper(img: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Blend image with a light yellow tint to simulate aged/paper look. Expects RGB."""
    # Light yellowish paper color (RGB)
    yellow_tint = np.array([245, 235, 200], dtype=np.float32)
    out = img.astype(np.float32)
    out = (1.0 - strength) * out + strength * yellow_tint
    return np.clip(out, 0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
# Render with NOTES header, TITLE BLOCK header, and larger fonts
# -----------------------------------------------------------------------------
def render_canvas_with_headers(
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
    """Same as render_canvas but adds NOTES header bar, TITLE BLOCK header bar, and larger fonts for notes/title text."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = BG_GREY
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = width / 3584.0
    fs = max(0.35, min(0.7, 0.45 * scale))
    th = max(1, int(round(scale)))
    line_w = max(1, min(3, int(round(scale))))
    border_lw = max(1, int(2 * scale))
    margin = sheet_margin

    # 1) Piping lines
    for line in lines:
        lw = line_w
        dashed = line["style"] == "dashed"
        for (xa, ya, xb, yb) in line["segments"]:
            draw_line_segment(img, xa, ya, xb, yb, LINE_COLOR, lw, dashed)

    # 2) Symbols
    for sym in placed:
        draw_symbol(img, sym, BG_GREY)

    # 3) Text (tags and line specs)
    for w in words:
        bbox = w["bbox"]
        x, y = bbox[0], bbox[3] - max(2, int(4 * scale))
        if x < 0 or y < 0 or y > height:
            continue
        txt = w["text"]
        cv2.putText(img, txt, (x, y), font, fs, TEXT_COLOR, th, cv2.LINE_AA)

    # 4) Outer dashed border
    ox1, oy1 = margin, margin
    ox2, oy2 = width - margin, height - margin
    draw_dashed_line(img, (ox1, oy1), (ox2, oy1), LINE_COLOR, border_lw)
    draw_dashed_line(img, (ox2, oy1), (ox2, oy2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (ox2, oy2), (ox1, oy2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (ox1, oy2), (ox1, oy1), LINE_COLOR, border_lw)

    # 5) Main diagram dashed border
    mx1, my1, mx2, my2 = regions["main_diagram"]
    draw_dashed_line(img, (mx1, my1), (mx2, my1), LINE_COLOR, border_lw)
    draw_dashed_line(img, (mx2, my1), (mx2, my2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (mx2, my2), (mx1, my2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (mx1, my2), (mx1, my1), LINE_COLOR, border_lw)

    # 6) Notes block — header bar + "NOTES" + larger body text
    nx1, ny1, nx2, ny2 = regions["notes"]
    draw_dashed_line(img, (nx1, ny1), (nx2, ny1), LINE_COLOR, border_lw)
    draw_dashed_line(img, (nx2, ny1), (nx2, ny2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (nx2, ny2), (nx1, ny2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (nx1, ny2), (nx1, ny1), LINE_COLOR, border_lw)

    note_pad = max(12, int(18 * scale))
    header_h = max(32, int(42 * scale))  # height of "NOTES" header bar
    cv2.rectangle(img, (nx1, ny1), (nx2, ny1 + header_h), (220, 220, 220), -1)  # light grey header bar
    cv2.line(img, (nx1, ny1 + header_h), (nx2, ny1 + header_h), LINE_COLOR, max(1, border_lw))

    notes_header_fs = max(0.55, 0.7 * scale)  # larger "NOTES" title
    notes_header_th = max(1, int(round(1.2 * scale)))
    cv2.putText(
        img, "NOTES",
        (nx1 + note_pad, ny1 + header_h - max(4, int(8 * scale))),
        font, notes_header_fs, TEXT_COLOR, notes_header_th, cv2.LINE_AA
    )

    note_fs = max(0.38, 0.44 * scale)  # larger notes body
    note_dy = ny1 + header_h + max(8, int(12 * scale))
    line_h = max(22, int(26 * scale))
    nlist = notes_list or REFERENCE_NOTES_28
    for i, text in enumerate(nlist[:28]):
        y_pos = note_dy + i * line_h
        if y_pos >= ny2 - note_pad:
            break
        txt = text[:75] + ".." if len(text) > 75 else text
        cv2.putText(img, txt, (nx1 + note_pad, y_pos), font, note_fs, TEXT_COLOR, th, cv2.LINE_AA)

    # 7) Title block — header bar + "TITLE BLOCK" + larger KeyValue/Table text
    tx1, ty1, tx2, ty2 = regions["title_block"]
    draw_dashed_line(img, (tx1, ty1), (tx2, ty1), LINE_COLOR, border_lw)
    draw_dashed_line(img, (tx2, ty1), (tx2, ty2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (tx2, ty2), (tx1, ty2), LINE_COLOR, border_lw)
    draw_dashed_line(img, (tx1, ty2), (tx1, ty1), LINE_COLOR, border_lw)

    title_pad = max(12, int(18 * scale))
    title_header_h = max(32, int(42 * scale))
    cv2.rectangle(img, (tx1, ty1), (tx2, ty1 + title_header_h), (220, 220, 220), -1)
    cv2.line(img, (tx1, ty1 + title_header_h), (tx2, ty1 + title_header_h), LINE_COLOR, max(1, border_lw))

    title_header_fs = max(0.5, 0.65 * scale)
    title_header_th = max(1, int(round(1.2 * scale)))
    cv2.putText(
        img, "TITLE BLOCK",
        (tx1 + title_pad, ty1 + title_header_h - max(4, int(8 * scale))),
        font, title_header_fs, TEXT_COLOR, title_header_th, cv2.LINE_AA
    )

    y_max = ty2 - title_pad
    y_off = ty1 + title_header_h + max(10, int(16 * scale))
    kv_fs = max(0.38, 0.44 * scale)  # larger KeyValue
    kv_line_h = max(22, int(26 * scale))
    for row in keyvalue:
        if y_off <= y_max:
            cv2.putText(img, f"{row[0]}: {row[1]}", (tx1 + title_pad, y_off), font, kv_fs, TEXT_COLOR, th, cv2.LINE_AA)
        y_off += kv_line_h
    y_off += int(10 * scale)
    tbl_fs = max(0.3, 0.36 * scale)  # larger table text
    tbl_line_h = max(20, int(24 * scale))
    for r in range(table.shape[0]):
        if y_off <= y_max:
            x_off = tx1 + title_pad
            for c in range(table.shape[1]):
                cv2.putText(img, str(table[r, c])[:14], (x_off, y_off), font, tbl_fs, TEXT_COLOR, th, cv2.LINE_AA)
                x_off += int(85 * scale)
        y_off += tbl_line_h

    return img


# -----------------------------------------------------------------------------
# Pipeline: render -> pixel augmentation + yellowish paper -> save
# -----------------------------------------------------------------------------
def generate_one(
    sample_id: int,
    loader: SymbolLoader,
    width: int,
    height: int,
    cfg: Dict[str, Any],
    pixel_processor: PixelLevelTransformProcessor,
    apply_yellow: bool = True,
) -> None:
    sheet_margin = max(80, int(90 * width / 3584))
    blocks_border_margin = max(60, int(70 * width / 3584))
    regions = canvas_regions(
        width,
        height,
        outer_margin=sheet_margin,
        right_col_pad=max(50, int(60 * width / 3584)),
        blocks_border_margin=blocks_border_margin,
    )
    main_rect = regions["main_diagram"]

    ref_kv, ref_tb, notes_list = load_reference_content(cfg["root"], sample_id, ref_width=width)
    keyvalue = ref_kv if ref_kv is not None else make_keyvalue(sample_id, width, height)
    table = ref_tb if ref_tb is not None else make_table()

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

    lines, linker_line = build_connections(
        placed,
        main_rect,
        connection_prob=0.65,
        distance_threshold=max(280, int(320 * width / 3584)),
        max_connections_per_symbol=4,
    )
    words, symbol_refs = build_words_and_linker(placed, lines, linker_line)

    canvas = render_canvas_with_headers(
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

    # BGR -> RGB for albumentations
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img_rgb = pixel_processor.apply_pixel_transforms(img_rgb)
    if apply_yellow:
        img_rgb = apply_yellowish_paper(img_rgb, strength=random.uniform(0.15, 0.35))
    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    out_img_dir = cfg["output_images_dir"]
    out_img_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_img_dir / f"{sample_id}.png"
    cv2.imwrite(str(img_path), out_bgr)

    gt_path = cfg["output_annotations_dir"] / f"gt_{sample_id}.txt"
    write_gt_txt(placed, gt_path, width, height)


def run(
    root: Optional[Path] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    num_images: int = 10,
    seed: Optional[int] = None,
    apply_yellowish_paper: bool = True,
) -> None:
    cfg = _config(root=root, width=width, height=height, num_images=num_images, seed=seed)
    if cfg["seed"] is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    w = cfg["width"]
    sym_max = 120 if w >= 5000 else 90
    sym_min = 50 if w >= 5000 else 36
    loader = SymbolLoader(cfg["classes_dir"], max_symbol_size=sym_max, min_symbol_size=sym_min)
    if not loader.symbols:
        raise FileNotFoundError(f"No symbols loaded from {cfg['classes_dir']}")

    pixel_processor = PixelLevelTransformProcessor()
    n = cfg["num_images"]
    print(f"DigitizePID augmented: {n} images, canvas {w}x{cfg['height']}, Output only (no ImagesInfo/.npy)")
    for i in range(n):
        generate_one(i, loader, w, cfg["height"], cfg, pixel_processor, apply_yellow=apply_yellowish_paper)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"  generated {i + 1}/{n}")
    print(f"Done. Images: {cfg['output_images_dir']}, Annotations: {cfg['output_annotations_dir']}")


if __name__ == "__main__":
    run(
        root=DEFAULT_ROOT,
        width=DEFAULT_CANVAS[0],
        height=DEFAULT_CANVAS[1],
        num_images=5,
        seed=42,
        apply_yellowish_paper=True,
    )
