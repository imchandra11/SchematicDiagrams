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
# KeyValue & Table
# -----------------------------------------------------------------------------
KEYVALUE_KEYS = [
    "PROJECT/LOCATION/ASSIGN", "ORGANIZATION", "CONTRACTOR PROJECT NO.",
    "CONTRACTOR NO.", "DRAWING NAME", "UNIT", "CONTRACT NAME", "SCALE",
    "DRAW/SHEET NO.", "REV",
]
TABLE_HEADER = ["ISSUE", "DATE", "MADE", "CHECK'D", "APPRV'D", "DESCRIPTION"]

# -----------------------------------------------------------------------------
# Reference Notes (28 items from DigitizePID sample)
# -----------------------------------------------------------------------------
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
# Connections
# -----------------------------------------------------------------------------
CONNECTION_PROB = 0.65
DISTANCE_THRESHOLD_BASE = 280
DISTANCE_THRESHOLD_SCALE = 320
MAX_CONNECTIONS_PER_SYMBOL = 4

# -----------------------------------------------------------------------------
# Symbol Sizing
# -----------------------------------------------------------------------------
SYMBOL_SIZE_LARGE_CANVAS_THRESHOLD = 5000
SYMBOL_MAX_LARGE = 120
SYMBOL_MAX_SMALL = 90
SYMBOL_MIN_LARGE = 50
SYMBOL_MIN_SMALL = 36
