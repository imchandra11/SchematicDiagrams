# Schematic Diagrams Dataset Generator for CRAFT Model

This project generates synthetic schematic diagram datasets for training the CRAFT (Character Region Awareness for Text Detection) model. The dataset generation process applies comprehensive spatial and pixel-level transformations to create realistic schematic diagrams with proper annotations.

## Overview

The dataset generation follows a multi-stage pipeline:
1. **White Background Creation** - Starts with a clean white background (255, 255, 255)
2. **Symbol Placement with Spatial Transforms** - Places instrument symbols with spatial-level augmentations
3. **Edge-to-Edge Connections** - Connects symbols with wires/lines (edge-to-edge, not center-to-center)
4. **Pixel-Level Transforms** - Applies realistic pixel-level augmentations to the entire image
5. **Annotation Generation** - Creates CRAFT ICDAR format annotations

## Dataset Generation Pipeline

### Stage 1: Symbol Spatial-Level Transforms

Spatial transforms are applied **individually to each symbol** before placement. Each symbol randomly receives 1-2 transforms from the following list:

#### Geometric Transforms

1. **Affine Transform** (p=0.7)
   - Scale: x-axis (0.8-1.2), y-axis (0.8-1.2)
   - Translation: ±10% in both directions
   - Rotation: -15° to +15°
   - Shear: -5° to +5°

2. **Rotate** (p=0.7)
   - Rotation limit: -15° to +15°
   - Border mode: Constant (white fill)
   - Fill color: (255, 255, 255) - matches background

3. **Scaling** (p=0.6)
   - Independent x and y scaling: 0.7x to 1.3x

4. **Transpose** (p=0.3)
   - Swaps width and height

5. **Horizontal Flip** (p=0.3)
   - Mirrors image horizontally

6. **Vertical Flip** (p=0.3)
   - Mirrors image vertically

7. **ShiftScaleRotate** (p=0.6)
   - Combined shift, scale, and rotation
   - Shift: ±10%
   - Scale: ±20%
   - Rotation: ±15°

#### Deformation Transforms

8. **GridDistortion** (p=0.4)
   - Grid steps: 5
   - Distortion limit: 0.3
   - Border mode: Constant (white fill)

9. **GridElasticDeform** (p=0.3)
   - Grid size: 4x4
   - Magnitude: 10 pixels
   - Creates elastic grid-based deformations

10. **PiecewiseAffine** (p=0.3)
    - Scale: 0.01 to 0.05
    - Grid: 4 rows × 4 columns
    - Creates piecewise affine transformations

11. **ThinPlateSpline** (p=0.3)
    - Scale range: 0.1 to 0.3
    - Control points: 4
    - Creates smooth deformations using thin plate spline interpolation

12. **ElasticTransform** (p=0.4)
    - Alpha: 50 (deformation strength)
    - Sigma: 5 (smoothness)
    - Creates elastic-like deformations

13. **OpticalDistortion** (p=0.3)
    - Distortion limit: 0.1
    - Simulates camera lens distortion

14. **SquareSymmetry** (p=0.2)
    - Applies square symmetry transformations

**Note**: All spatial transforms use white fill color (255, 255, 255) to match the background, ensuring seamless integration when symbols are rotated or deformed.

### Stage 2: Symbol Placement

- Symbols are randomly selected (5-20 per image)
- Random positions with collision avoidance
- Random rotation angles (-15° to +15°) with proper padding
- Symbols are resized to 50-150px (maintaining aspect ratio)

### Stage 3: Edge-to-Edge Connections

- Symbols are connected with wires/lines **edge-to-edge** (not center-to-center)
- Connection probability: 30-50% for symbols within 400-600px distance
- Line widths: Random (2-5 pixels)
- Line colors: Dark gray to black (0-50 intensity)
- Finds closest edge points between symbols for realistic connections

### Stage 4: Pixel-Level Transforms

Pixel-level transforms are applied to the **entire image** after symbol placement and connections. Multiple transforms are randomly selected from different categories:

#### Noise-Based Augmentations

1. **GaussNoise** (p=0.5)
   - Standard deviation range: 0.1 to 0.2
   - Mean range: 0.0 to 0.0
   - Adds Gaussian noise

2. **ISONoise** (p=0.4)
   - Color shift: 0.01 to 0.05
   - Intensity: 0.1 to 0.5
   - Simulates ISO camera noise

3. **MultiplicativeNoise** (p=0.4)
   - Multiplier: 0.9 to 1.1
   - Per-channel: Yes
   - Multiplicative noise pattern

4. **SaltAndPepper** (p=0.3)
   - Amount: 1% to 5% of pixels
   - Salt vs Pepper ratio: 40% to 60%
   - Impulse noise

5. **ShotNoise** (p=0.3)
   - Simulates photon counting noise (Poisson process)

#### Blur-Based Augmentations

6. **Blur** (p=0.5)
   - Blur limit: 3 to 7 pixels

7. **GaussianBlur** (p=0.5)
   - Blur limit: 3 to 7 pixels
   - Gaussian kernel blur

8. **MotionBlur** (p=0.3)
   - Blur limit: 7 pixels
   - Simulates motion blur

9. **MedianBlur** (p=0.3)
   - Blur limit: 5 pixels
   - Median filter blur

10. **GlassBlur** (p=0.2)
    - Sigma: 0.7
    - Max delta: 4
    - Iterations: 2
    - Glass-like distortion blur

#### Color & Lighting Adjustments

11. **ColorJitter** (p=0.6)
    - Brightness: ±20%
    - Contrast: ±20%
    - Saturation: ±20%
    - Hue: ±10%

12. **CLAHE** (p=0.4)
    - Clip limit: 2.0
    - Tile grid size: 8×8
    - Contrast Limited Adaptive Histogram Equalization

13. **RandomGamma** (p=0.4)
    - Gamma limit: 80 to 120
    - Gamma correction

14. **RandomBrightnessContrast** (p=0.5)
    - Brightness limit: ±20%
    - Contrast limit: ±20%

15. **HueSaturationValue** (p=0.4)
    - Hue shift: ±10
    - Saturation shift: ±15
    - Value shift: ±10

#### Weather & Environmental Effects

16. **RandomFog** (p=0.2)
    - Fog coefficient range: 0.3 to 0.5
    - Alpha coefficient: 0.1
    - Simulates fog effect

17. **RandomSnow** (p=0.2)
    - Snow point range: 0.1 to 0.3
    - Brightness coefficient: 2.5
    - Simulates snow effect

18. **RandomRain** (p=0.2)
    - Slant range: -10° to +10°
    - Drop length: 20 pixels
    - Drop width: 1 pixel
    - Drop color: (200, 200, 200)
    - Blur value: 3
    - Brightness coefficient: 0.7

19. **RandomShadow** (p=0.2)
    - Shadow ROI: (0, 0.5, 1, 1) - bottom half of image
    - Number of shadows: 1 to 2
    - Shadow dimension: 5 vertices
    - Simulates shadow effects

#### Artistic & Stylization Effects

20. **Sharpen** (p=0.3)
    - Alpha: 0.2 to 0.5
    - Lightness: 0.5 to 1.0
    - Sharpening filter

21. **Posterize** (p=0.2)
    - Number of bits: 4 to 8
    - Reduces color depth

22. **ToSepia** (p=0.2)
    - Converts to sepia tone

23. **Emboss** (p=0.2)
    - Alpha: 0.2 to 0.5
    - Strength: 0.2 to 0.7
    - Emboss effect

#### Camera & Sensor Simulation

24. **PixelDropout** (p=0.2)
    - Dropout probability: 1%
    - Random pixel dropout

#### Special Effects & Textures

25. **Spatter** (p=0.2)
    - Intensity: 0.1 to 0.3
    - Color: (200, 200, 200)
    - Simulates spatter/water drops

## Output Format

### Image Format
- **Format**: PNG
- **Dimensions**: 1920×1080 (landscape)
- **Color Space**: RGB
- **Location**: `output/images/schematic_XXX.png`

### Annotation Format (CRAFT ICDAR)
- **Format**: Text file (.txt)
- **Location**: `output/annotations/schematic_XXX.txt`
- **Format**: `x1,y1,x2,y2,x3,y3,x4,y4,transcription`
  - Coordinates: 4-point polygon (top-left, top-right, bottom-right, bottom-left)
  - Transcription: Symbol image name without extension (e.g., "Flow_Controller")

**Example**:
```
828,559,950,559,950,684,828,684,Indicator_3
1471,263,1547,263,1547,345,1471,345,Shared_Indicator_2
```

## Usage

### Prerequisites

Install required packages:
```bash
pip install -r requirements.txt
```

### Generate Dataset

```bash
python generate_realistic_dataset.py
```

### Configuration

Edit the main function at the bottom of `generate_realistic_dataset.py`:

```python
generate_realistic_dataset(
    output_dir='output',      # Output directory
    num_images=30,            # Number of images to generate
    symbols_dir='Instruments', # Directory containing symbol images
    width=1920,               # Image width
    height=1080                # Image height
)
```

## Key Features

1. **Realistic Transformations**: Comprehensive spatial and pixel-level transforms create natural-looking variations
2. **Edge-to-Edge Connections**: Symbols are connected realistically at their edges, not centers
3. **Background Matching**: All transforms use white fill (255, 255, 255) to match background
4. **Robust Error Handling**: Gracefully handles transform failures and corrupted images
5. **CRAFT Compatible**: Output format matches CRAFT model training requirements

## Transform Selection Logic

- **Spatial Transforms**: 1-2 transforms randomly selected per symbol
- **Pixel Transforms**: Multiple transforms randomly selected from different categories:
  - Noise: 70% chance
  - Blur: 60% chance
  - Color: 70% chance
  - Weather: 30% chance
  - Artistic: 30% chance
  - Camera: 20% chance
  - Special: 20% chance

## Notes

- All transforms use probability values (p) to control application frequency
- Spatial transforms are applied before symbol placement
- Pixel transforms are applied after all symbols and connections are placed
- Rotation padding uses background color to prevent clipping
- Symbols maintain aspect ratio during resizing

## Requirements

- Python 3.7+
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- albumentations >= 1.3.0
- Pillow >= 10.0.0
