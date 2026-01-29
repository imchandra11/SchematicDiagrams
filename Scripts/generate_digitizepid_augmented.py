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
    generate_random_notes,
    generate_random_table_header,
    generate_random_keyvalue_keys,
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
# Render with NOTES header, TITLE BLOCK header, and larger fonts; variable layout
# -----------------------------------------------------------------------------
def _draw_text_horizontal(img: np.ndarray, txt: str, x: int, y: int, font: int, fs: float, th: int) -> None:
    """Draw text horizontally (baseline at y)."""
    cv2.putText(img, txt, (x, y), font, fs, TEXT_COLOR, th, cv2.LINE_AA)


def _draw_text_rotated_90(
    img: np.ndarray,
    txt: str,
    anchor_x: int,
    anchor_y: int,
    font: int,
    fs: float,
    th: int,
) -> None:
    """Draw text rotated 90° anticlockwise, no background: only text pixels are drawn (transparent)."""
    # Key color = transparent; only pixels not equal to key are copied (text only)
    KEY_BGR = (255, 0, 255)  # BGR magenta; text is black
    (tw, th_tex), baseline = cv2.getTextSize(txt, font, fs, th)
    pad = max(4, th_tex // 2)
    pw = tw + pad * 2
    ph = th_tex + baseline + pad * 2
    patch = np.empty((ph, pw, 3), dtype=np.uint8)
    patch[:] = KEY_BGR
    cv2.putText(patch, txt, (pad, pad + th_tex), font, fs, TEXT_COLOR, th, cv2.LINE_AA)
    # 90° anticlockwise via cv2.rotate (no blending); result shape (pw, ph) -> (ph, pw)
    rotated = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rw = rotated.shape[1]
    rh = rotated.shape[0]
    x1 = anchor_x - rw // 2
    y1 = anchor_y - rh // 2
    x2 = x1 + rw
    y2 = y1 + rh
    ih, iw = img.shape[:2]
    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = rw - max(0, x2 - iw)
    sy2 = rh - max(0, y2 - ih)
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(iw, x2)
    y2_clip = min(ih, y2)
    if x2_clip <= x1_clip or y2_clip <= y1_clip:
        return
    roi = img[y1_clip:y2_clip, x1_clip:x2_clip]
    src = rotated[sy1:sy2, sx1:sx2]
    if src.shape[0] != roi.shape[0] or src.shape[1] != roi.shape[1]:
        return
    # Copy only text pixels (not key color) so no background is drawn
    not_key = (src[:, :, 0] != KEY_BGR[0]) | (src[:, :, 1] != KEY_BGR[1]) | (src[:, :, 2] != KEY_BGR[2])
    roi[not_key] = src[not_key]


def render_canvas_with_headers(
    width: int,
    height: int,
    placed: List[Dict],
    lines: List[Dict],
    words: List[Dict],
    keyvalue: np.ndarray,
    table: np.ndarray,
    regions: Dict[str, Optional[Tuple[int, int, int, int]]],
    notes_list: Optional[List[str]] = None,
    sheet_margin: int = 40,
    has_notes: bool = True,
    has_title_block: bool = True,
    has_table: bool = True,
    has_description_block: bool = False,
    description_lines: Optional[List[str]] = None,
) -> np.ndarray:
    """Render canvas with optional NOTES, DESCRIPTION, TITLE BLOCK. Line specs: horizontal on horizontal segments, vertical on vertical segments."""
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

    # 3) Text (tags and line specs); line specs at path midpoint with margin from line; vertical = rotated 90°
    for w in words:
        bbox = w["bbox"]
        txt = w["text"]
        if w.get("line_spec_vertical"):
            anchor = w.get("line_spec_anchor")
            if anchor is None:
                anchor = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            px, py = anchor
            if px < 0 or py < 0 or px > width or py > height:
                continue
            _draw_text_rotated_90(img, txt, px, py, font, fs, th)
        elif w.get("line_spec_anchor"):
            # Horizontal line spec: centered at anchor (already offset from line)
            px, py = w["line_spec_anchor"]
            (tw, th_tex), baseline = cv2.getTextSize(txt, font, fs, th)
            x = px - tw // 2
            y = py + th_tex // 2
            if x < 0 or y < 0 or y > height or x + tw > width:
                continue
            _draw_text_horizontal(img, txt, x, y, font, fs, th)
        else:
            # Tag (symbol label)
            x, y = bbox[0], bbox[3] - max(2, int(4 * scale))
            if x < 0 or y < 0 or y > height:
                continue
            _draw_text_horizontal(img, txt, x, y, font, fs, th)

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

    # 6) Notes block — only if has_notes
    if has_notes and regions.get("notes"):
        nx1, ny1, nx2, ny2 = regions["notes"]
        draw_dashed_line(img, (nx1, ny1), (nx2, ny1), LINE_COLOR, border_lw)
        draw_dashed_line(img, (nx2, ny1), (nx2, ny2), LINE_COLOR, border_lw)
        draw_dashed_line(img, (nx2, ny2), (nx1, ny2), LINE_COLOR, border_lw)
        draw_dashed_line(img, (nx1, ny2), (nx1, ny1), LINE_COLOR, border_lw)
        note_pad = max(12, int(18 * scale))
        header_h = max(32, int(42 * scale))
        cv2.rectangle(img, (nx1, ny1), (nx2, ny1 + header_h), (220, 220, 220), -1)
        cv2.line(img, (nx1, ny1 + header_h), (nx2, ny1 + header_h), LINE_COLOR, max(1, border_lw))
        notes_header_fs = max(0.55, 0.7 * scale)
        notes_header_th = max(1, int(round(1.2 * scale)))
        cv2.putText(img, "NOTES", (nx1 + note_pad, ny1 + header_h - max(4, int(8 * scale))), font, notes_header_fs, TEXT_COLOR, notes_header_th, cv2.LINE_AA)
        note_fs = max(0.38, 0.44 * scale)
        note_dy = ny1 + header_h + max(8, int(12 * scale))
        line_h = max(22, int(26 * scale))
        nlist = notes_list or []
        for i, text in enumerate(nlist[:28]):
            y_pos = note_dy + i * line_h
            if y_pos >= ny2 - note_pad:
                break
            txt = text[:75] + ".." if len(text) > 75 else text
            cv2.putText(img, txt, (nx1 + note_pad, y_pos), font, note_fs, TEXT_COLOR, th, cv2.LINE_AA)

    # 6b) Description block — only if has_description_block and region exists
    if has_description_block and description_lines and regions.get("description"):
        dx1, dy1, dx2, dy2 = regions["description"]
        draw_dashed_line(img, (dx1, dy1), (dx2, dy1), LINE_COLOR, border_lw)
        draw_dashed_line(img, (dx2, dy1), (dx2, dy2), LINE_COLOR, border_lw)
        draw_dashed_line(img, (dx2, dy2), (dx1, dy2), LINE_COLOR, border_lw)
        draw_dashed_line(img, (dx1, dy2), (dx1, dy1), LINE_COLOR, border_lw)
        desc_pad = max(12, int(18 * scale))
        desc_header_h = max(28, int(36 * scale))
        cv2.rectangle(img, (dx1, dy1), (dx2, dy1 + desc_header_h), (230, 230, 230), -1)
        cv2.line(img, (dx1, dy1 + desc_header_h), (dx2, dy1 + desc_header_h), LINE_COLOR, max(1, border_lw))
        cv2.putText(img, "DESCRIPTION", (dx1 + desc_pad, dy1 + desc_header_h - max(4, int(6 * scale))), font, max(0.45, 0.55 * scale), TEXT_COLOR, th, cv2.LINE_AA)
        desc_fs = max(0.32, 0.38 * scale)
        desc_line_h = max(18, int(22 * scale))
        y_desc = dy1 + desc_header_h + max(6, int(10 * scale))
        for i, text in enumerate(description_lines):
            if y_desc >= dy2 - desc_pad:
                break
            txt = (text[:70] + "..") if len(text) > 70 else text
            cv2.putText(img, txt, (dx1 + desc_pad, y_desc), font, desc_fs, TEXT_COLOR, th, cv2.LINE_AA)
            y_desc += desc_line_h

    # 7) Title block — only if has_title_block
    if has_title_block and regions.get("title_block"):
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
        cv2.putText(img, "TITLE BLOCK", (tx1 + title_pad, ty1 + title_header_h - max(4, int(8 * scale))), font, title_header_fs, TEXT_COLOR, title_header_th, cv2.LINE_AA)
        y_max = ty2 - title_pad
        y_off = ty1 + title_header_h + max(10, int(16 * scale))
        kv_fs = max(0.38, 0.44 * scale)
        kv_line_h = max(22, int(26 * scale))
        for row in keyvalue:
            if y_off <= y_max:
                cv2.putText(img, f"{row[0]}: {row[1]}", (tx1 + title_pad, y_off), font, kv_fs, TEXT_COLOR, th, cv2.LINE_AA)
            y_off += kv_line_h
        if has_table and table.size > 0:
            y_off += int(10 * scale)
            tbl_fs = max(0.3, 0.36 * scale)
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

    # Variable layout: some images notes only, some title only, some both; optional table; optional DESCRIPTION block
    has_notes = random.random() < 0.85
    has_title_block = random.random() < 0.85
    has_table = (random.random() < 0.7) and has_title_block
    has_description_block = random.random() < 0.5

    regions = canvas_regions(
        width,
        height,
        outer_margin=sheet_margin,
        right_col_pad=max(RIGHT_COL_PAD_BASE, int(RIGHT_COL_PAD_SCALE * width / REF_WIDTH)),
        blocks_border_margin=blocks_border_margin,
        has_description_block=has_description_block,
    )
    main_rect = regions["main_diagram"]

    ref_kv, ref_tb, notes_list = load_reference_content(cfg["root"], sample_id, ref_width=REF_WIDTH_FULL)
    if ref_kv is None:
        keyvalue = make_keyvalue(sample_id, width, height, keys=generate_random_keyvalue_keys())
    else:
        keyvalue = ref_kv
    if ref_tb is None:
        table = make_table(header=generate_random_table_header()) if has_table else np.array([[]], dtype=object)
    else:
        table = ref_tb if has_table else np.array([[]], dtype=object)
    if not notes_list:
        notes_list = generate_random_notes()

    description_lines = generate_random_notes(count=random.randint(3, 10)) if has_description_block else None

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
        notes_list=notes_list if has_notes else None,
        sheet_margin=sheet_margin,
        has_notes=has_notes,
        has_title_block=has_title_block,
        has_table=has_table and table.size > 0,
        has_description_block=has_description_block and bool(description_lines),
        description_lines=description_lines,
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
