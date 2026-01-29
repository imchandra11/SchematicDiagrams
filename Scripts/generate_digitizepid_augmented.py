"""
DigitizePID augmented P&ID dataset generator.

Reuses logic from generate_digitizepid.py and adds:
- Pixel-level augmentation (blur, noise, environmental effects) from generate_enhanced_dataset.py
- Yellowish paper effect
- NOTES header bar and TITLE BLOCK header bar (like reference images)
- Larger font sizes for notes and title-block text

Output only:
- output_dir/Images/0.png, 1.png, ...
- output_dir/Annotations/gt_0.txt, gt_1.txt, ...

Does NOT save ImagesInfo or any .npy files.
"""

import cv2
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from constants import (
    DEFAULT_CANVAS,
    BG_GREY,
    LINE_COLOR,
    TEXT_COLOR,
    REFERENCE_NOTES_28,
    REF_WIDTH,
    REF_WIDTH_FULL,
    SHEET_MARGIN_BASE,
    SHEET_MARGIN_SCALE,
    BLOCKS_BORDER_MARGIN_BASE,
    BLOCKS_BORDER_MARGIN_SCALE,
    RIGHT_COL_PAD_BASE,
    RIGHT_COL_PAD_SCALE,
    SYMBOL_COUNT_MIN,
    SYMBOL_COUNT_RANGE,
    GRID_STEP_BASE,
    GRID_STEP_SCALE,
    GRID_STEP_Y_RATIO,
    CONNECTION_PROB,
    DISTANCE_THRESHOLD_BASE,
    DISTANCE_THRESHOLD_SCALE,
    SYMBOL_SIZE_LARGE_CANVAS_THRESHOLD,
    SYMBOL_MAX_LARGE,
    SYMBOL_MAX_SMALL,
    SYMBOL_MIN_LARGE,
    SYMBOL_MIN_SMALL,
    YELLOW_TINT_RGB,
    YELLOWISH_PAPER_STRENGTH_MIN,
    YELLOWISH_PAPER_STRENGTH_MAX,
)

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
    SymbolLoader,
    _grid_cells,
)

from generate_enhanced_dataset import PixelLevelTransformProcessor


def _config(
    output_dir: Path,
    symbols_dir: Path,
    num_images: int = 10,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build config from output_dir, symbols_dir, num_images."""
    output_dir = Path(output_dir)
    symbols_dir = Path(symbols_dir)
    w = width or DEFAULT_CANVAS[0]
    h = height or DEFAULT_CANVAS[1]
    # Root for reference content (KeyValue, Table, Notes) - parent of symbols_dir
    root = symbols_dir.parent
    return {
        "root": root,
        "width": w,
        "height": h,
        "num_images": num_images,
        "seed": seed,
        "classes_dir": symbols_dir,
        "output_images_dir": output_dir / "Images",
        "output_annotations_dir": output_dir / "Annotations",
    }


# -----------------------------------------------------------------------------
# Yellowish paper effect (RGB image)
# -----------------------------------------------------------------------------
def apply_yellowish_paper(img: np.ndarray, strength: float = 0.25) -> np.ndarray:
    """Blend image with a light yellow tint to simulate aged/paper look. Expects RGB."""
    yellow_tint = np.array(YELLOW_TINT_RGB, dtype=np.float32)
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
    scale = width / REF_WIDTH
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
    sheet_margin = max(SHEET_MARGIN_BASE, int(SHEET_MARGIN_SCALE * width / REF_WIDTH))
    blocks_border_margin = max(BLOCKS_BORDER_MARGIN_BASE, int(BLOCKS_BORDER_MARGIN_SCALE * width / REF_WIDTH))
    regions = canvas_regions(
        width,
        height,
        outer_margin=sheet_margin,
        right_col_pad=max(RIGHT_COL_PAD_BASE, int(RIGHT_COL_PAD_SCALE * width / REF_WIDTH)),
        blocks_border_margin=blocks_border_margin,
    )
    main_rect = regions["main_diagram"]

    ref_kv, ref_tb, notes_list = load_reference_content(cfg["root"], sample_id, ref_width=REF_WIDTH_FULL)
    keyvalue = ref_kv if ref_kv is not None else make_keyvalue(sample_id, width, height)
    table = ref_tb if ref_tb is not None else make_table()

    step = max(GRID_STEP_BASE, int(GRID_STEP_SCALE * width / REF_WIDTH))
    cell_count = len(_grid_cells(main_rect, step_x=step, step_y=int(step * GRID_STEP_Y_RATIO), jitter=0))
    num_sym = min(cell_count, max(SYMBOL_COUNT_MIN, random.randint(*SYMBOL_COUNT_RANGE)))
    symbols = loader.get_random_symbols(num_sym)
    placed = place_symbols(
        symbols,
        main_rect,
        use_grid=True,
        grid_step_x=step,
        grid_step_y=int(step * GRID_STEP_Y_RATIO),
    )
    if len(placed) == 0:
        print(f"  [sample {sample_id}] no symbols placed, skip")
        return

    lines, linker_line = build_connections(
        placed,
        main_rect,
        connection_prob=CONNECTION_PROB,
        distance_threshold=max(DISTANCE_THRESHOLD_BASE, int(DISTANCE_THRESHOLD_SCALE * width / REF_WIDTH)),
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
        img_rgb = apply_yellowish_paper(img_rgb, strength=random.uniform(YELLOWISH_PAPER_STRENGTH_MIN, YELLOWISH_PAPER_STRENGTH_MAX))
    out_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    out_img_dir = cfg["output_images_dir"]
    out_img_dir.mkdir(parents=True, exist_ok=True)
    img_path = out_img_dir / f"{sample_id}.png"
    cv2.imwrite(str(img_path), out_bgr)

    gt_path = cfg["output_annotations_dir"] / f"gt_{sample_id}.txt"
    write_gt_txt(placed, gt_path, width, height)


def run(
    output_dir: str,
    symbols_dir: str,
    num_images: int = 10,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    apply_yellowish_paper: bool = True,
) -> None:
    """
    Generate DigitizePID augmented dataset.

    Args:
        output_dir: Absolute path to output directory (Images and Annotations subdirs will be created).
        symbols_dir: Absolute path to symbols/classes directory (e.g., DigitizePID_Dataset/Classes).
        num_images: Number of images to generate.
        width: Canvas width (default: 7168).
        height: Canvas height (default: 4561).
        seed: Random seed for reproducibility.
        apply_yellowish_paper: Whether to apply yellowish paper effect.
    """
    cfg = _config(
        output_dir=Path(output_dir),
        symbols_dir=Path(symbols_dir),
        num_images=num_images,
        width=width,
        height=height,
        seed=seed,
    )
    if cfg["seed"] is not None:
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

    w = cfg["width"]
    sym_max = SYMBOL_MAX_LARGE if w >= SYMBOL_SIZE_LARGE_CANVAS_THRESHOLD else SYMBOL_MAX_SMALL
    sym_min = SYMBOL_MIN_LARGE if w >= SYMBOL_SIZE_LARGE_CANVAS_THRESHOLD else SYMBOL_MIN_SMALL
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
    import argparse
    parser = argparse.ArgumentParser(description="DigitizePID augmented dataset generator")
    parser.add_argument("--output_dir", required=True, help="Absolute path to output directory")
    parser.add_argument("--symbols_dir", required=True, help="Absolute path to symbols/classes directory")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate")
    parser.add_argument("--width", type=int, default=None, help="Canvas width (default: 7168)")
    parser.add_argument("--height", type=int, default=None, help="Canvas height (default: 4561)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no_yellowish", action="store_true", help="Disable yellowish paper effect")
    args = parser.parse_args()

    run(
        output_dir=args.output_dir,
        symbols_dir=args.symbols_dir,
        num_images=args.num_images,
        width=args.width,
        height=args.height,
        seed=args.seed,
        apply_yellowish_paper=not args.no_yellowish,
    )
