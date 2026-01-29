"""
Constants for DigitizePID dataset generation.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Paths & Defaults
# -----------------------------------------------------------------------------
DEFAULT_ROOT = Path(__file__).resolve().parent / "DigitizePID_Dataset"
DEFAULT_WIDTH = 7168
DEFAULT_HEIGHT = 4561
DEFAULT_CANVAS = (7168, 4561)

# Reference dimensions for scaling
REF_WIDTH = 3584
REF_HEIGHT = 4561
REF_WIDTH_FULL = 7168

# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
BG_GREY = (204, 204, 204)
LINE_COLOR = (0, 0, 0)
TEXT_COLOR = (0, 0, 0)

# Yellowish paper effect
YELLOW_TINT_RGB = (245, 235, 200)
YELLOWISH_PAPER_STRENGTH_MIN = 0.15
YELLOWISH_PAPER_STRENGTH_MAX = 0.35

# -----------------------------------------------------------------------------
# Layout & Margins
# -----------------------------------------------------------------------------
SHEET_MARGIN_BASE = 80
SHEET_MARGIN_SCALE = 90
BLOCKS_BORDER_MARGIN_BASE = 60
BLOCKS_BORDER_MARGIN_SCALE = 70
RIGHT_COL_PAD_BASE = 50
RIGHT_COL_PAD_SCALE = 60

# -----------------------------------------------------------------------------
# Symbol Placement
# -----------------------------------------------------------------------------
SYMBOL_COUNT_MIN = 30
SYMBOL_COUNT_RANGE = (50, 90)
GRID_STEP_BASE = 80
GRID_STEP_SCALE = 100
GRID_STEP_Y_RATIO = 0.9

# -----------------------------------------------------------------------------
# Connections & Line specs
# -----------------------------------------------------------------------------
CONNECTION_PROB = 0.65
DISTANCE_THRESHOLD_BASE = 280
DISTANCE_THRESHOLD_SCALE = 320
MAX_CONNECTIONS_PER_SYMBOL = 4
# Clear gap (pixels) between the line and the nearest edge of line-spec text
LINE_SPEC_MARGIN = 180
LINE_SPEC_MARGIN_MIN = 120
# For vertically rotated specs only: extra margin so they stay clearly off the vertical line
LINE_SPEC_VERTICAL_MARGIN_EXTRA = 200

# -----------------------------------------------------------------------------
# Symbol Sizing (slightly larger for visibility)
# -----------------------------------------------------------------------------
SYMBOL_SIZE_LARGE_CANVAS_THRESHOLD = 5000
SYMBOL_MAX_LARGE = 132
SYMBOL_MAX_SMALL = 100
SYMBOL_MIN_LARGE = 55
SYMBOL_MIN_SMALL = 40
