import cv2
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import albumentations as A


class SpatialTransformProcessor:
    """Apply spatial-level transforms on symbol images without shear, maintaining aspect ratio."""
    
    def __init__(self):
        # Spatial transforms for symbols (without shear)
        self.spatial_transforms = [
            # Affine transforms (no shear)
            lambda: A.Affine(
                scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},  # Reduced to prevent clipping
                rotate=(-15, 15),
                shear=0,  # No shear
                p=0.7
            ),
            # Rotation
            lambda: A.Rotate(limit=(-15, 15), border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.7),
            # Uniform scaling (maintains aspect ratio)
            lambda: A.Affine(scale={'x': (0.85, 1.15), 'y': (0.85, 1.15)}, p=0.6),
            # Transpose
            lambda: A.Transpose(p=0.3),
            # Horizontal flip
            lambda: A.HorizontalFlip(p=0.3),
            # Vertical flip
            lambda: A.VerticalFlip(p=0.3),
            # Shift scale rotate (reduced limits)
            lambda: A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.15, rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.6
            ),
        ]
    
    def apply_spatial_transform(self, img: np.ndarray, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Apply random spatial transform to symbol image with aspect ratio preservation."""
        # Select random transform(s)
        num_transforms = random.randint(1, 2)
        selected_transforms = random.sample(self.spatial_transforms, num_transforms)
        
        # Create transform pipeline
        transform_list = []
        for transform_func in selected_transforms:
            transform = transform_func()
            # Update border fill value to match background (RGB tuple)
            if hasattr(transform, 'fill'):
                if isinstance(bg_color, tuple) and len(bg_color) == 3:
                    transform.fill = bg_color
                elif hasattr(bg_color, '__getitem__') and len(bg_color) == 3:
                    transform.fill = tuple(bg_color)
                else:
                    transform.fill = bg_color[0] if hasattr(bg_color, '__getitem__') else bg_color
            
            if hasattr(transform, 'fill_mask'):
                if isinstance(bg_color, tuple) and len(bg_color) == 3:
                    transform.fill_mask = bg_color
                elif hasattr(bg_color, '__getitem__') and len(bg_color) == 3:
                    transform.fill_mask = tuple(bg_color)
            transform_list.append(transform)
        
        # Apply transforms with error handling
        if transform_list:
            try:
                transform_pipeline = A.Compose(transform_list)
                result = transform_pipeline(image=img)['image']
                # Validate result
                if result is None or result.size == 0 or result.shape[0] == 0 or result.shape[1] == 0:
                    result = img
            except Exception as e:
                print(f"Warning: Spatial transform failed: {e}, using original image")
                result = img
        else:
            result = img
        
        return result


class PixelLevelTransformProcessor:
    """Apply pixel-level transforms on entire image with higher probability for dirty images."""
    
    def __init__(self):
        # Noise-based augmentations (increased probability)
        self.noise_transforms = [
            lambda: A.GaussNoise(std_range=(0.15, 0.3), mean_range=(0.0, 0.0), p=0.7),
            lambda: A.ISONoise(color_shift=(0.01, 0.08), intensity=(0.15, 0.6), p=0.6),
            lambda: A.MultiplicativeNoise(multiplier=(0.85, 1.15), per_channel=True, p=0.5),
            lambda: A.SaltAndPepper(amount=(0.02, 0.08), salt_vs_pepper=(0.4, 0.6), p=0.4),
            lambda: A.ShotNoise(p=0.4),
        ]
        
        # Blur-based augmentations (increased probability)
        self.blur_transforms = [
            lambda: A.Blur(blur_limit=(3, 9), p=0.6),
            lambda: A.GaussianBlur(blur_limit=(3, 9), p=0.6),
            lambda: A.MotionBlur(blur_limit=9, p=0.4),
            lambda: A.MedianBlur(blur_limit=7, p=0.4),
            lambda: A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=0.3),
        ]
        
        # Color & Lighting adjustments (increased probability)
        self.color_transforms = [
            lambda: A.ColorJitter(
                brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1, p=0.7
            ),
            lambda: A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            lambda: A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            lambda: A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.6
            ),
            lambda: A.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=0.5
            ),
        ]
        
        # Weather & Environmental effects (increased probability)
        self.weather_transforms = [
            lambda: A.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.1, p=0.3),
            lambda: A.RandomSnow(
                snow_point_range=(0.1, 0.3), brightness_coeff=2.5, p=0.3
            ),
            lambda: A.RandomRain(
                slant_range=(-10, 10), drop_length=20, drop_width=1,
                drop_color=(200, 200, 200), blur_value=3, brightness_coefficient=0.7, p=0.3
            ),
            lambda: A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2),
                shadow_dimension=5, p=0.3
            ),
        ]
        
        # Artistic & Stylization effects (increased probability)
        self.artistic_transforms = [
            lambda: A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.4),
            lambda: A.Posterize(num_bits=(4, 8), p=0.3),
            lambda: A.ToSepia(p=0.3),
            lambda: A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.3),
        ]
        
        # Camera & Sensor simulation (increased probability)
        self.camera_transforms = [
            lambda: A.PixelDropout(dropout_prob=0.015, p=0.3),
        ]
        
        # Special effects & textures (increased probability)
        self.special_transforms = [
            lambda: A.Spatter(
                intensity=(0.15, 0.4), color=(200, 200, 200), p=0.3
            ),
        ]
    
    def apply_pixel_transforms(self, img: np.ndarray) -> np.ndarray:
        """Apply random pixel-level transforms with higher probability for dirty images."""
        transform_list = []
        
        # Increased probabilities for more dirty/noisy images
        if random.random() < 0.85:  # Increased from 0.7
            transform_list.append(random.choice(self.noise_transforms)())
        
        if random.random() < 0.75:  # Increased from 0.6
            transform_list.append(random.choice(self.blur_transforms)())
        
        if random.random() < 0.8:  # Increased from 0.7
            transform_list.append(random.choice(self.color_transforms)())
        
        if random.random() < 0.5:  # Increased from 0.3
            transform_list.append(random.choice(self.weather_transforms)())
        
        if random.random() < 0.5:  # Increased from 0.3
            transform_list.append(random.choice(self.artistic_transforms)())
        
        if random.random() < 0.4:  # Increased from 0.2
            transform_list.append(random.choice(self.camera_transforms)())
        
        if random.random() < 0.4:  # Increased from 0.2
            transform_list.append(random.choice(self.special_transforms)())
        
        # Apply transforms
        if transform_list:
            transform_pipeline = A.Compose(transform_list)
            result = transform_pipeline(image=img)['image']
        else:
            result = img
        
        return result


def add_grid_overlay(img: np.ndarray, grid_spacing: int = 50, line_color: Tuple[int, int, int] = (200, 200, 200),
                     line_thickness: int = 1, alpha: float = 0.3) -> np.ndarray:
    """Add a grid overlay to the image."""
    result = img.copy()
    h, w = img.shape[:2]
    
    # Create grid overlay
    overlay = result.copy()
    
    # Draw vertical lines
    for x in range(0, w, grid_spacing):
        cv2.line(overlay, (x, 0), (x, h), line_color, line_thickness)
    
    # Draw horizontal lines
    for y in range(0, h, grid_spacing):
        cv2.line(overlay, (0, y), (w, y), line_color, line_thickness)
    
    # Blend overlay with original image
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    
    return result


def add_black_border(img: np.ndarray, border_width: int = 5) -> np.ndarray:
    """Add black border around the image."""
    h, w = img.shape[:2]
    result = np.zeros((h + 2 * border_width, w + 2 * border_width, 3), dtype=np.uint8)
    result[border_width:h + border_width, border_width:w + border_width] = img
    return result


class SymbolLoader:
    """Load and process symbol images."""
    
    def __init__(self, symbols_dir: str):
        self.symbols_dir = symbols_dir
        self.symbol_paths = list(Path(symbols_dir).glob('*.png'))
        if not self.symbol_paths:
            raise ValueError(f"No PNG files found in {symbols_dir}")
    
    def load_symbol(self, symbol_path: Path) -> Tuple[np.ndarray, str]:
        """Load a symbol image and return image and label."""
        img = cv2.imread(str(symbol_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Could not load image: {symbol_path}")
        
        # Handle transparency
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            bg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            alpha_3d = alpha[:, :, np.newaxis] / 255.0
            img = (img * alpha_3d + bg * (1 - alpha_3d)).astype(np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = symbol_path.stem
        return img, label
    
    def resize_symbol(self, img: np.ndarray, min_size: int = 50, max_size: int = 150) -> np.ndarray:
        """Resize symbol to random size while maintaining aspect ratio."""
        h, w = img.shape[:2]
        target_size = random.randint(min_size, max_size)
        
        # Maintain aspect ratio
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    def get_random_symbol(self) -> Tuple[np.ndarray, str]:
        """Get a random symbol."""
        symbol_path = random.choice(self.symbol_paths)
        img, label = self.load_symbol(symbol_path)
        img = self.resize_symbol(img)
        return img, label


class SchematicComposer:
    """Compose schematic diagrams with enhanced edge-to-edge connections."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
    
    def get_mid_edge_points(self, x: int, y: int, img: np.ndarray, angle: float = 0) -> List[Tuple[int, int]]:
        """Get mid-point of each edge (top, right, bottom, left) for connections."""
        h, w = img.shape[:2]
        
        # Mid-points of edges relative to center
        mid_edges = [
            (0, -h/2),      # top mid
            (w/2, 0),       # right mid
            (0, h/2),       # bottom mid
            (-w/2, 0)       # left mid
        ]
        
        # Rotate if needed
        if angle != 0:
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            mid_edges = np.array(mid_edges)
            mid_edges = mid_edges @ rotation_matrix.T
            mid_edges = [(int(e[0]), int(e[1])) for e in mid_edges]
        else:
            mid_edges = [(int(e[0]), int(e[1])) for e in mid_edges]
        
        # Translate to actual position
        edge_points = [(x + e[0], y + e[1]) for e in mid_edges]
        return edge_points
    
    def get_bounding_box(self, x: int, y: int, img: np.ndarray, angle: float = 0) -> List[Tuple[int, int]]:
        """Get 4-point bounding box polygon coordinates."""
        h, w = img.shape[:2]
        
        corners = np.array([
            [-w/2, -h/2],
            [w/2, -h/2],
            [w/2, h/2],
            [-w/2, h/2]
        ])
        
        if angle != 0:
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners = corners @ rotation_matrix.T
        
        corners[:, 0] += x
        corners[:, 1] += y
        
        corners_list = [(int(c[0]), int(c[1])) for c in corners]
        sorted_by_y = sorted(corners_list, key=lambda p: (p[1], p[0]))
        top_two = sorted_by_y[:2]
        bottom_two = sorted_by_y[2:]
        
        top_left = min(top_two, key=lambda p: p[0])
        top_right = max(top_two, key=lambda p: p[0])
        bottom_left = min(bottom_two, key=lambda p: p[0])
        bottom_right = max(bottom_two, key=lambda p: p[0])
        
        return [top_left, top_right, bottom_right, bottom_left]
    
    def check_collision(self, x: int, y: int, img: np.ndarray, 
                       placed_symbols: List[Dict], padding: int = 30) -> bool:
        """Check if placement would cause collision."""
        h, w = img.shape[:2]
        new_rect = (x - w//2 - padding, y - h//2 - padding, 
                   x + w//2 + padding, y + h//2 + padding)
        
        for symbol in placed_symbols:
            sym_x, sym_y = symbol['x'], symbol['y']
            sym_img = symbol['image']
            sym_h, sym_w = sym_img.shape[:2]
            sym_rect = (sym_x - sym_w//2 - padding, sym_y - sym_h//2 - padding,
                       sym_x + sym_w//2 + padding, sym_y + sym_h//2 + padding)
            
            if not (new_rect[2] < sym_rect[0] or new_rect[0] > sym_rect[2] or
                   new_rect[3] < sym_rect[1] or new_rect[1] > sym_rect[3]):
                return True
        return False
    
    def place_symbols(self, background: np.ndarray, symbols: List[Tuple[np.ndarray, str]],
                     spatial_processor: SpatialTransformProcessor,
                     num_symbols: int = None, max_attempts: int = 150) -> Tuple[np.ndarray, List[Dict]]:
        """Place symbols on background with spatial transforms, ensuring no clipping."""
        if num_symbols is None:
            num_symbols = random.randint(15, 20)
        
        num_symbols = min(num_symbols, len(symbols))
        selected_symbols = random.sample(symbols, num_symbols)
        
        canvas = background.copy()
        placed_symbols = []
        bg_color = (255, 255, 255)
        
        for img, label in selected_symbols:
            # Apply spatial transform
            img_transformed = spatial_processor.apply_spatial_transform(img, bg_color)
            
            # Random rotation angle
            angle = random.uniform(-15, 15)
            
            h, w = img_transformed.shape[:2]
            if h == 0 or w == 0:
                continue
            
            # Calculate rotated dimensions with extra padding to prevent clipping
            if angle != 0:
                rad = math.radians(abs(angle))
                cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
                rot_w = int(h * sin_a + w * cos_a) + 10  # Extra padding
                rot_h = int(h * cos_a + w * sin_a) + 10
                rot_w = max(rot_w, 10)
                rot_h = max(rot_h, 10)
            else:
                rot_w, rot_h = w, h
            
            # Find placement position (with larger margin to prevent clipping)
            margin = max(rot_w, rot_h) // 2 + 80
            attempts = 0
            while attempts < max_attempts:
                x = random.randint(margin, self.width - margin)
                y = random.randint(margin, self.height - margin)
                
                temp_img = np.zeros((rot_h, rot_w, 3), dtype=np.uint8)
                if not self.check_collision(x, y, temp_img, placed_symbols):
                    break
                attempts += 1
            
            if attempts >= max_attempts:
                continue
            
            if img_transformed is None or img_transformed.size == 0:
                continue
            
            # Rotate with padding to prevent clipping
            if angle != 0 and rot_w > 0 and rot_h > 0 and w > 0 and h > 0:
                try:
                    if rot_w > 5000 or rot_h > 5000:
                        img_rotated = img_transformed
                        angle = 0
                    else:
                        padded_img = np.ones((rot_h, rot_w, 3), dtype=np.uint8) * bg_color
                        offset_x = (rot_w - w) // 2
                        offset_y = (rot_h - h) // 2
                        
                        if offset_y >= 0 and offset_x >= 0 and offset_y + h <= rot_h and offset_x + w <= rot_w:
                            padded_img[offset_y:offset_y + h, offset_x:offset_x + w] = img_transformed
                        
                        center = (rot_w / 2.0, rot_h / 2.0)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img_rotated = cv2.warpAffine(padded_img, rotation_matrix, (rot_w, rot_h),
                                                    borderValue=bg_color, flags=cv2.INTER_LINEAR)
                except Exception as e:
                    img_rotated = img_transformed
                    angle = 0
            else:
                img_rotated = img_transformed
                if angle != 0:
                    angle = 0
            
            # Place symbol on canvas
            h_rot, w_rot = img_rotated.shape[:2]
            x1 = max(0, x - w_rot//2)
            y1 = max(0, y - h_rot//2)
            x2 = min(self.width, x + w_rot//2)
            y2 = min(self.height, y + h_rot//2)
            
            sym_x1 = max(0, w_rot//2 - x)
            sym_y1 = max(0, h_rot//2 - y)
            sym_x2 = sym_x1 + (x2 - x1)
            sym_y2 = sym_y1 + (y2 - y1)
            
            canvas[y1:y2, x1:x2] = img_rotated[sym_y1:sym_y2, sym_x1:sym_x2]
            
            placed_symbols.append({
                'x': x,
                'y': y,
                'image': img_transformed,
                'label': label,
                'angle': angle,
                'rotated_image': img_rotated
            })
        
        return canvas, placed_symbols
    
    def line_segment_intersects_polygon(self, pt1: Tuple[int, int], pt2: Tuple[int, int],
                                       polygon: List[Tuple[int, int]]) -> bool:
        """Check if a line segment intersects with a polygon."""
        # Check if line segment intersects any edge of the polygon
        for i in range(len(polygon)):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % len(polygon)]
            
            if self.segments_intersect(pt1, pt2, p1, p2):
                return True
        
        return False
    
    def segments_intersect(self, p1: Tuple[int, int], p2: Tuple[int, int],
                          p3: Tuple[int, int], p4: Tuple[int, int]) -> bool:
        """Check if two line segments intersect using orientation method."""
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            if val == 0:
                return 0  # Collinear
            return 1 if val > 0 else 2  # Clockwise or counterclockwise
        
        def on_segment(p, q, r):
            return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                   q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))
        
        o1 = orientation(p1, p2, p3)
        o2 = orientation(p1, p2, p4)
        o3 = orientation(p3, p4, p1)
        o4 = orientation(p3, p4, p2)
        
        # General case
        if o1 != o2 and o3 != o4:
            return True
        
        # Special cases - collinear segments
        if o1 == 0 and on_segment(p1, p3, p2):
            return True
        if o2 == 0 and on_segment(p1, p4, p2):
            return True
        if o3 == 0 and on_segment(p3, p1, p4):
            return True
        if o4 == 0 and on_segment(p3, p2, p4):
            return True
        
        return False
    
    def path_crosses_symbol(self, pt1: Tuple[int, int], pt2: Tuple[int, int],
                           placed_symbols: List[Dict], exclude_indices: Tuple[int, int]) -> bool:
        """Check if a line segment crosses any symbol (excluding the two being connected)."""
        for idx, symbol in enumerate(placed_symbols):
            # Skip the two symbols being connected
            if idx == exclude_indices[0] or idx == exclude_indices[1]:
                continue
            
            # Get bounding box polygon for this symbol
            x, y = symbol['x'], symbol['y']
            img = symbol['image']
            angle = symbol['angle']
            polygon = self.get_bounding_box(x, y, img, angle)
            
            # Check if line segment intersects with this polygon
            if self.line_segment_intersects_polygon(pt1, pt2, polygon):
                return True
        
        return False
    
    def path_crosses_symbol_3_segment(self, pt1: Tuple[int, int], pt2: Tuple[int, int],
                                     placed_symbols: List[Dict], exclude_indices: Tuple[int, int],
                                     use_v_h_v: bool = None) -> bool:
        """Check if a 3-segment path (V-H-V or H-V-H) crosses any symbol."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Determine segment points
        if use_v_h_v is None:
            use_v_h_v = random.random() < 0.5
        
        if use_v_h_v:
            # Vertical-Horizontal-Vertical
            mid1_x = x1
            mid1_y = (y1 + y2) // 2
            mid2_x = x2
            mid2_y = mid1_y
        else:
            # Horizontal-Vertical-Horizontal
            mid1_x = (x1 + x2) // 2
            mid1_y = y1
            mid2_x = mid1_x
            mid2_y = y2
        
        # Check each segment
        if self.path_crosses_symbol((x1, y1), (mid1_x, mid1_y), placed_symbols, exclude_indices):
            return True
        if self.path_crosses_symbol((mid1_x, mid1_y), (mid2_x, mid2_y), placed_symbols, exclude_indices):
            return True
        if self.path_crosses_symbol((mid2_x, mid2_y), (x2, y2), placed_symbols, exclude_indices):
            return True
        
        return False
    
    def draw_3_segment_connection(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                                 line_color: Tuple[int, int, int], line_width: int,
                                 use_v_h_v: bool = None) -> np.ndarray:
        """Draw 3-segment connection: V-H-V or H-V-H."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Randomly choose V-H-V or H-V-H if not specified
        if use_v_h_v is None:
            use_v_h_v = random.random() < 0.5
        
        if use_v_h_v:
            # Vertical-Horizontal-Vertical
            mid1_x = x1
            mid1_y = (y1 + y2) // 2
            mid2_x = x2
            mid2_y = mid1_y
        else:
            # Horizontal-Vertical-Horizontal
            mid1_x = (x1 + x2) // 2
            mid1_y = y1
            mid2_x = mid1_x
            mid2_y = y2
        
        # Draw first segment
        cv2.line(img, (x1, y1), (mid1_x, mid1_y), line_color, line_width)
        # Draw middle segment
        cv2.line(img, (mid1_x, mid1_y), (mid2_x, mid2_y), line_color, line_width)
        # Draw final segment
        cv2.line(img, (mid2_x, mid2_y), (x2, y2), line_color, line_width)
        
        return img
    
    def connect_symbols_edge_to_edge(self, canvas: np.ndarray, placed_symbols: List[Dict],
                                     connection_prob: float = 0.4, 
                                     distance_threshold: int = 500) -> np.ndarray:
        """Connect symbols with wires. Each symbol has 1-3 connections using mid-edge points.
        Lines will not cross over other symbols."""
        result = canvas.copy()
        
        if len(placed_symbols) < 2:
            return result
        
        connection_count = {i: 0 for i in range(len(placed_symbols))}
        connections_made = []
        
        # Build list of potential connections
        potential_connections = []
        for i, sym1 in enumerate(placed_symbols):
            for j, sym2 in enumerate(placed_symbols[i+1:], start=i+1):
                dist = math.sqrt((sym1['x'] - sym2['x'])**2 + (sym1['y'] - sym2['y'])**2)
                if dist < distance_threshold:
                    # Get mid-edge points for both symbols
                    edges1 = self.get_mid_edge_points(sym1['x'], sym1['y'], sym1['image'], sym1['angle'])
                    edges2 = self.get_mid_edge_points(sym2['x'], sym2['y'], sym2['image'], sym2['angle'])
                    
                    # Find closest mid-edge points that don't cross other symbols
                    min_dist = float('inf')
                    best_edge1 = None
                    best_edge2 = None
                    best_use_3_segment = None
                    best_path_found = False
                    
                    # Try both straight line and 3-segment connections
                    for e1 in edges1:
                        for e2 in edges2:
                            d = math.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2)
                            
                            # Check straight line
                            if not self.path_crosses_symbol(e1, e2, placed_symbols, (i, j)):
                                if d < min_dist:
                                    min_dist = d
                                    best_edge1 = e1
                                    best_edge2 = e2
                                    best_use_3_segment = False
                                    best_path_found = True
                            
                            # Check 3-segment V-H-V
                            if not self.path_crosses_symbol_3_segment(e1, e2, placed_symbols, (i, j), use_v_h_v=True):
                                if d < min_dist:
                                    min_dist = d
                                    best_edge1 = e1
                                    best_edge2 = e2
                                    best_use_3_segment = True  # Will use V-H-V
                                    best_path_found = True
                            
                            # Check 3-segment H-V-H
                            if not self.path_crosses_symbol_3_segment(e1, e2, placed_symbols, (i, j), use_v_h_v=False):
                                if d < min_dist:
                                    min_dist = d
                                    best_edge1 = e1
                                    best_edge2 = e2
                                    best_use_3_segment = True  # Will use H-V-H, but we'll pick randomly later
                                    best_path_found = True
                    
                    if best_edge1 and best_edge2 and best_path_found:
                        potential_connections.append({
                            'i': i,
                            'j': j,
                            'dist': dist,
                            'edge1': best_edge1,
                            'edge2': best_edge2,
                            'use_3_segment': best_use_3_segment if best_use_3_segment else False,
                            'path_found': True
                        })
        
        # Sort by distance
        potential_connections.sort(key=lambda x: x['dist'])
        
        # Phase 1: Ensure each symbol has at least 1 connection
        for conn in potential_connections:
            i, j = conn['i'], conn['j']
            if connection_count[i] == 0 or connection_count[j] == 0:
                # Double-check path doesn't cross (safety check)
                edge1 = conn['edge1']
                edge2 = conn['edge2']
                if conn['use_3_segment']:
                    # For 3-segment, at least one pattern should not cross
                    crosses_v_h_v = self.path_crosses_symbol_3_segment(edge1, edge2, placed_symbols, (i, j), True)
                    crosses_h_v_h = self.path_crosses_symbol_3_segment(edge1, edge2, placed_symbols, (i, j), False)
                    if not crosses_v_h_v or not crosses_h_v_h:
                        connections_made.append((i, j))
                        connection_count[i] += 1
                        connection_count[j] += 1
                else:
                    if not self.path_crosses_symbol(edge1, edge2, placed_symbols, (i, j)):
                        connections_made.append((i, j))
                        connection_count[i] += 1
                        connection_count[j] += 1
        
        # Phase 2: Add more connections up to max 3 per symbol
        for conn in potential_connections:
            i, j = conn['i'], conn['j']
            if (i, j) in connections_made:
                continue
            if connection_count[i] >= 3 or connection_count[j] >= 3:
                continue
            if random.random() < connection_prob:
                # Double-check path doesn't cross (safety check)
                edge1 = conn['edge1']
                edge2 = conn['edge2']
                if conn['use_3_segment']:
                    # For 3-segment, at least one pattern should not cross
                    crosses_v_h_v = self.path_crosses_symbol_3_segment(edge1, edge2, placed_symbols, (i, j), True)
                    crosses_h_v_h = self.path_crosses_symbol_3_segment(edge1, edge2, placed_symbols, (i, j), False)
                    if not crosses_v_h_v or not crosses_h_v_h:
                        connections_made.append((i, j))
                        connection_count[i] += 1
                        connection_count[j] += 1
                else:
                    if not self.path_crosses_symbol(edge1, edge2, placed_symbols, (i, j)):
                        connections_made.append((i, j))
                        connection_count[i] += 1
                        connection_count[j] += 1
        
        # Draw all connections
        for conn in potential_connections:
            i, j = conn['i'], conn['j']
            if (i, j) in connections_made:
                edge1 = conn['edge1']
                edge2 = conn['edge2']
                
                line_width = random.randint(1, 2)
                line_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
                
                # Use 3-segment if marked, otherwise use straight line (or try 3-segment randomly)
                use_3_segment = conn.get('use_3_segment', False)
                if not use_3_segment:
                    use_3_segment = random.random() < 0.6  # 60% chance for 3-segment when path allows
                
                if use_3_segment:
                    # Check which pattern doesn't cross and use that one
                    crosses_v_h_v = self.path_crosses_symbol_3_segment(edge1, edge2, placed_symbols, (i, j), True)
                    crosses_h_v_h = self.path_crosses_symbol_3_segment(edge1, edge2, placed_symbols, (i, j), False)
                    
                    if not crosses_v_h_v and not crosses_h_v_h:
                        # Both work, randomly choose
                        use_v_h_v = random.random() < 0.5
                    elif not crosses_v_h_v:
                        # Only V-H-V works
                        use_v_h_v = True
                    elif not crosses_h_v_h:
                        # Only H-V-H works
                        use_v_h_v = False
                    else:
                        # Both cross (shouldn't happen if logic is correct), use straight line as fallback
                        use_3_segment = False
                    
                    if use_3_segment:
                        result = self.draw_3_segment_connection(
                            result,
                            (int(edge1[0]), int(edge1[1])),
                            (int(edge2[0]), int(edge2[1])),
                            line_color,
                            line_width,
                            use_v_h_v=use_v_h_v
                        )
                    else:
                        cv2.line(result,
                                (int(edge1[0]), int(edge1[1])),
                                (int(edge2[0]), int(edge2[1])),
                                line_color, line_width)
                else:
                    cv2.line(result,
                            (int(edge1[0]), int(edge1[1])),
                            (int(edge2[0]), int(edge2[1])),
                            line_color, line_width)
        
        return result
    
    def extract_polygon_coordinates(self, placed_symbols: List[Dict]) -> List[Dict]:
        """Extract 4-point polygon coordinates for each placed symbol."""
        annotations = []
        
        for symbol in placed_symbols:
            x, y = symbol['x'], symbol['y']
            img = symbol['image']
            angle = symbol['angle']
            
            polygon = self.get_bounding_box(x, y, img, angle)
            
            polygon = [
                (max(0, min(self.width-1, p[0])), max(0, min(self.height-1, p[1])))
                for p in polygon
            ]
            
            annotations.append({
                'polygon': polygon,
                'label': symbol['label']
            })
        
        return annotations


class AnnotationWriter:
    """Write CRAFT ICDAR format annotations."""
    
    @staticmethod
    def write_annotation(filepath: str, annotations: List[Dict]):
        """Write annotation file in CRAFT ICDAR format."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for ann in annotations:
                polygon = ann['polygon']
                label = ann['label']
                
                coords = ','.join([f"{p[0]},{p[1]}" for p in polygon])
                line = f"{coords},{label}\n"
                f.write(line)


def generate_enhanced_dataset(output_dir: str = 'output', num_images: int = 100,
                              symbols_dir: str = 'Instruments', width: int = 1920, height: int = 1080):
    """Generate enhanced realistic schematic diagram dataset with all improvements."""
    
    # Create output directories
    images_dir = Path(output_dir) / 'images'
    annotations_dir = Path(output_dir) / 'annotations'
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    symbol_loader = SymbolLoader(symbols_dir)
    spatial_processor = SpatialTransformProcessor()
    pixel_processor = PixelLevelTransformProcessor()
    composer = SchematicComposer(width, height)
    annotation_writer = AnnotationWriter()
    
    print(f"Loaded {len(symbol_loader.symbol_paths)} symbols")
    print(f"Generating {num_images} enhanced schematic diagrams...")
    
    for i in range(num_images):
        # Create white background
        background = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Prepare symbols for this image (15-20 symbols)
        symbols_for_image = []
        num_symbols = random.randint(15, 20)
        for _ in range(num_symbols):
            sym_img, sym_label = symbol_loader.get_random_symbol()
            symbols_for_image.append((sym_img, sym_label))
        
        # Place symbols with spatial transforms
        canvas, placed_symbols = composer.place_symbols(
            background, symbols_for_image, spatial_processor, num_symbols=num_symbols
        )
        
        # Connect symbols edge-to-edge
        canvas = composer.connect_symbols_edge_to_edge(
            canvas, placed_symbols,
            connection_prob=random.uniform(0.3, 0.5),
            distance_threshold=random.randint(400, 600)
        )
        
        # Add grid to some images (30% chance)
        if random.random() < 0.3:
            grid_spacing = random.choice([40, 50, 60])
            canvas = add_grid_overlay(canvas, grid_spacing=grid_spacing, alpha=0.25)
        
        # Apply pixel-level transforms (higher probability for dirty images)
        canvas = pixel_processor.apply_pixel_transforms(canvas)
        
        # Add black border around schematic
        canvas = add_black_border(canvas, border_width=10)
        
        # Update width and height for annotations (after border)
        border_width = 10
        actual_width = width
        actual_height = height
        
        # Extract polygon coordinates (before border was added, so use original dimensions)
        # We need to adjust coordinates for border offset
        annotations = composer.extract_polygon_coordinates(placed_symbols)
        
        # Adjust polygon coordinates for border offset
        adjusted_annotations = []
        for ann in annotations:
            adjusted_polygon = [(p[0] + border_width, p[1] + border_width) for p in ann['polygon']]
            adjusted_annotations.append({
                'polygon': adjusted_polygon,
                'label': ann['label']
            })
        
        # Save image
        image_filename = f"schematic_{i+1:03d}.png"
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        # Save annotation
        annotation_filename = f"schematic_{i+1:03d}.txt"
        annotation_path = annotations_dir / annotation_filename
        annotation_writer.write_annotation(str(annotation_path), adjusted_annotations)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_images} images")
    
    print(f"\nDataset generation complete!")
    print(f"Images saved to: {images_dir}")
    print(f"Annotations saved to: {annotations_dir}")


if __name__ == '__main__':
    generate_enhanced_dataset(
        output_dir='output',
        num_images=50,
        symbols_dir='Instruments',
        width=1920,
        height=1080
    )
