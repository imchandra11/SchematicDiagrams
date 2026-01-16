import cv2
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import albumentations as A
# Note: Importing specific transforms not needed, using A.TransformName directly


class SpatialTransformProcessor:
    """Apply spatial-level transforms on symbol images."""
    
    def __init__(self):
        # Spatial transforms for symbols
        self.spatial_transforms = [
            # Affine transforms
            lambda: A.Affine(
                scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                rotate=(-15, 15),
                shear=(-5, 5),
                p=0.7
            ),
            # Rotation
            lambda: A.Rotate(limit=(-15, 15), border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.7),
            # Scaling
            lambda: A.Affine(scale={'x': (0.7, 1.3), 'y': (0.7, 1.3)}, p=0.6),
            # Transpose
            lambda: A.Transpose(p=0.3),
            # Horizontal flip
            lambda: A.HorizontalFlip(p=0.3),
            # Vertical flip
            lambda: A.VerticalFlip(p=0.3),
            # Shift scale rotate
            lambda: A.ShiftScaleRotate(
                shift_limit=0.1, scale_limit=0.2, rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.6
            ),
            # # Grid distortion
            # lambda: A.GridDistortion(
            #     num_steps=5, distort_limit=0.3,
            #     border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.4
            # ),
            # # Grid elastic deform (doesn't support fill, uses border_mode only)
            # lambda: A.GridElasticDeform(
            #     num_grid_xy=(4, 4), magnitude=10,
            #     p=0.3
            # ),
            # # Piecewise affine
            # lambda: A.PiecewiseAffine(
            #     scale=(0.01, 0.05), nb_rows=4, nb_cols=4,
            #     border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.3
            # ),
            # # Thin plate spline
            # lambda: A.ThinPlateSpline(
            #     scale_range=(0.1, 0.3), num_control_points=4,
            #     border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.3
            # ),
            # # Elastic transform
            # lambda: A.ElasticTransform(
            #     alpha=50, sigma=5,
            #     border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.4
            # ),
            # # Optical distortion
            # lambda: A.OpticalDistortion(
            #     distort_limit=0.1,
            #     border_mode=cv2.BORDER_CONSTANT, fill=(255, 255, 255), p=0.3
            # ),
            # # Square symmetry
            # lambda: A.SquareSymmetry(p=0.2),
        ]
    
    def apply_spatial_transform(self, img: np.ndarray, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Apply random spatial transform to symbol image."""
        # Select random transform(s)
        num_transforms = random.randint(1, 2)
        selected_transforms = random.sample(self.spatial_transforms, num_transforms)
        
        # Create transform pipeline
        transform_list = []
        for transform_func in selected_transforms:
            transform = transform_func()
            # Update border fill value to match background (RGB tuple)
            if hasattr(transform, 'fill'):
                # For RGB images, fill should be a tuple (R, G, B)
                if isinstance(bg_color, tuple) and len(bg_color) == 3:
                    transform.fill = bg_color  # Use full RGB tuple (255, 255, 255)
                elif hasattr(bg_color, '__getitem__') and len(bg_color) == 3:
                    transform.fill = tuple(bg_color)  # Convert to tuple
                else:
                    # Fallback: use single value (applies to all channels)
                    transform.fill = bg_color[0] if hasattr(bg_color, '__getitem__') else bg_color
            
            # Also check for fill_mask attribute (for some transforms)
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
    """Apply pixel-level transforms on entire image."""
    
    def __init__(self):
        # Noise-based augmentations
        self.noise_transforms = [
            lambda: A.GaussNoise(std_range=(0.1, 0.2), mean_range=(0.0, 0.0), p=0.5),
            lambda: A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.4),
            lambda: A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, p=0.4),
            lambda: A.SaltAndPepper(amount=(0.01, 0.05), salt_vs_pepper=(0.4, 0.6), p=0.3),
            lambda: A.ShotNoise(p=0.3),
        ]
        
        # Blur-based augmentations
        self.blur_transforms = [
            lambda: A.Blur(blur_limit=(3, 7), p=0.5),
            lambda: A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            lambda: A.MotionBlur(blur_limit=7, p=0.3),
            lambda: A.MedianBlur(blur_limit=5, p=0.3),
            lambda: A.GlassBlur(sigma=0.7, max_delta=4, iterations=2, p=0.2),
        ]
        
        # Color & Lighting adjustments
        self.color_transforms = [
            lambda: A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.6
            ),
            lambda: A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
            lambda: A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            lambda: A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            lambda: A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4
            ),
        ]
        
        # Weather & Environmental effects
        self.weather_transforms = [
            lambda: A.RandomFog(fog_coef_range=(0.3, 0.5), alpha_coef=0.1, p=0.2),
            lambda: A.RandomSnow(
                snow_point_range=(0.1, 0.3), brightness_coeff=2.5, p=0.2
            ),
            lambda: A.RandomRain(
                slant_range=(-10, 10), drop_length=20, drop_width=1,
                drop_color=(200, 200, 200), blur_value=3, brightness_coefficient=0.7, p=0.2
            ),
            lambda: A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1), num_shadows_limit=(1, 2),
                shadow_dimension=5, p=0.2
            ),
        ]
        
        # Artistic & Stylization effects
        self.artistic_transforms = [
            lambda: A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
            lambda: A.Posterize(num_bits=(4, 8), p=0.2),
            lambda: A.ToSepia(p=0.2),
            lambda: A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=0.2),
        ]
        
        # Camera & Sensor simulation
        self.camera_transforms = [
            lambda: A.PixelDropout(dropout_prob=0.01, p=0.2),
        ]
        
        # Special effects & textures
        self.special_transforms = [
            lambda: A.Spatter(
                intensity=(0.1, 0.3), color=(200, 200, 200), p=0.2
            ),
        ]
    
    def apply_pixel_transforms(self, img: np.ndarray) -> np.ndarray:
        """Apply random pixel-level transforms to entire image."""
        transform_list = []
        
        # Randomly select transforms from each category
        if random.random() < 0.7:
            transform_list.append(random.choice(self.noise_transforms)())
        
        if random.random() < 0.6:
            transform_list.append(random.choice(self.blur_transforms)())
        
        if random.random() < 0.7:
            transform_list.append(random.choice(self.color_transforms)())
        
        if random.random() < 0.3:
            transform_list.append(random.choice(self.weather_transforms)())
        
        if random.random() < 0.3:
            transform_list.append(random.choice(self.artistic_transforms)())
        
        if random.random() < 0.2:
            transform_list.append(random.choice(self.camera_transforms)())
        
        if random.random() < 0.2:
            transform_list.append(random.choice(self.special_transforms)())
        
        # Apply transforms
        if transform_list:
            transform_pipeline = A.Compose(transform_list)
            result = transform_pipeline(image=img)['image']
        else:
            result = img
        
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
            # Extract alpha channel
            alpha = img[:, :, 3]
            # Convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            # Create white background
            bg = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
            alpha_3d = alpha[:, :, np.newaxis] / 255.0
            img = (img * alpha_3d + bg * (1 - alpha_3d)).astype(np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = symbol_path.stem  # Filename without extension
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
    """Compose schematic diagrams with edge-to-edge connections."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
    
    def get_edge_points(self, x: int, y: int, img: np.ndarray, angle: float = 0) -> List[Tuple[int, int]]:
        """Get edge connection points (top, right, bottom, left) for a symbol."""
        h, w = img.shape[:2]
        
        # Edge points relative to center
        edges = [
            (0, -h/2),      # top
            (w/2, 0),       # right
            (0, h/2),       # bottom
            (-w/2, 0)       # left
        ]
        
        # Rotate if needed
        if angle != 0:
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            edges = np.array(edges)
            edges = edges @ rotation_matrix.T
            edges = [(int(e[0]), int(e[1])) for e in edges]
        else:
            edges = [(int(e[0]), int(e[1])) for e in edges]
        
        # Translate to actual position
        edge_points = [(x + e[0], y + e[1]) for e in edges]
        return edge_points
    
    def get_bounding_box(self, x: int, y: int, img: np.ndarray, angle: float = 0) -> List[Tuple[int, int]]:
        """Get 4-point bounding box polygon coordinates."""
        h, w = img.shape[:2]
        
        # Corner points relative to center
        corners = np.array([
            [-w/2, -h/2],  # top-left
            [w/2, -h/2],   # top-right
            [w/2, h/2],    # bottom-right
            [-w/2, h/2]    # bottom-left
        ])
        
        # Rotate if needed
        if angle != 0:
            rad = math.radians(angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners = corners @ rotation_matrix.T
        
        # Translate to actual position
        corners[:, 0] += x
        corners[:, 1] += y
        
        # Convert to list of tuples and sort
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
                     num_symbols: int = None, max_attempts: int = 100) -> Tuple[np.ndarray, List[Dict]]:
        """Place symbols on white background with spatial transforms."""
        if num_symbols is None:
            num_symbols = random.randint(15, 20)
        
        num_symbols = min(num_symbols, len(symbols))
        selected_symbols = random.sample(symbols, num_symbols)
        
        canvas = background.copy()
        placed_symbols = []
        bg_color = (255, 255, 255)  # White background
        
        for img, label in selected_symbols:
            # Apply spatial transform
            img_transformed = spatial_processor.apply_spatial_transform(img, bg_color)
            
            # Random rotation angle
            angle = random.uniform(-15, 15)
            
            # Calculate rotated dimensions
            h, w = img_transformed.shape[:2]
            # Ensure minimum dimensions
            if h == 0 or w == 0:
                continue
            
            if angle != 0:
                rad = math.radians(abs(angle))
                cos_a, sin_a = abs(math.cos(rad)), abs(math.sin(rad))
                rot_w = int(h * sin_a + w * cos_a)
                rot_h = int(h * cos_a + w * sin_a)
                # Ensure minimum dimensions
                rot_w = max(rot_w, 10)
                rot_h = max(rot_h, 10)
            else:
                rot_w, rot_h = w, h
            
            # Find placement position
            attempts = 0
            while attempts < max_attempts:
                x = random.randint(rot_w//2 + 50, self.width - rot_w//2 - 50)
                y = random.randint(rot_h//2 + 50, self.height - rot_h//2 - 50)
                
                temp_img = np.zeros((rot_h, rot_w, 3), dtype=np.uint8)
                if not self.check_collision(x, y, temp_img, placed_symbols):
                    break
                attempts += 1
            
            if attempts >= max_attempts:
                continue
            
            # Validate transformed image before rotation
            if img_transformed is None or img_transformed.size == 0:
                continue
            
            # Rotate if needed (with background color padding)
            if angle != 0 and rot_w > 0 and rot_h > 0 and w > 0 and h > 0:
                try:
                    # Ensure dimensions are reasonable
                    if rot_w > 5000 or rot_h > 5000:
                        img_rotated = img_transformed
                        angle = 0
                    else:
                        padded_img = np.ones((rot_h, rot_w, 3), dtype=np.uint8) * bg_color
                        offset_x = (rot_w - w) // 2
                        offset_y = (rot_h - h) // 2
                        
                        # Ensure valid indices
                        if offset_y >= 0 and offset_x >= 0 and offset_y + h <= rot_h and offset_x + w <= rot_w:
                            padded_img[offset_y:offset_y + h, offset_x:offset_x + w] = img_transformed
                        
                        center = (rot_w / 2.0, rot_h / 2.0)
                        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                        img_rotated = cv2.warpAffine(padded_img, rotation_matrix, (rot_w, rot_h),
                                                    borderValue=bg_color, flags=cv2.INTER_LINEAR)
                except Exception as e:
                    # If rotation fails, use original image
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
            
            # Store symbol info
            placed_symbols.append({
                'x': x,
                'y': y,
                'image': img_transformed,  # Original transformed image for polygon
                'label': label,
                'angle': angle,
                'rotated_image': img_rotated
            })
        
        return canvas, placed_symbols
    
    def draw_l_shaped_connection(self, img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int],
                                 line_color: Tuple[int, int, int], line_width: int) -> np.ndarray:
        """Draw L-shaped connection (horizontal then vertical or vice versa)."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Randomly choose horizontal-first or vertical-first
        if random.random() < 0.5:
            # Horizontal then vertical
            mid_x = x2
            mid_y = y1
        else:
            # Vertical then horizontal
            mid_x = x1
            mid_y = y2
        
        # Draw horizontal segment
        cv2.line(img, (x1, y1), (mid_x, mid_y), line_color, line_width)
        # Draw vertical segment
        cv2.line(img, (mid_x, mid_y), (x2, y2), line_color, line_width)
        
        return img
    
    def connect_symbols_edge_to_edge(self, canvas: np.ndarray, placed_symbols: List[Dict],
                                     connection_prob: float = 0.4, 
                                     distance_threshold: int = 500) -> np.ndarray:
        """Connect symbols with wires/lines edge-to-edge. Each symbol has 1-3 connections."""
        result = canvas.copy()
        
        if len(placed_symbols) < 2:
            return result
        
        # Track connections per symbol (by index)
        connection_count = {i: 0 for i in range(len(placed_symbols))}
        connections_made = []  # List of (i, j) pairs that are connected
        
        # Build list of potential connections with distances
        potential_connections = []
        for i, sym1 in enumerate(placed_symbols):
            for j, sym2 in enumerate(placed_symbols[i+1:], start=i+1):
                dist = math.sqrt((sym1['x'] - sym2['x'])**2 + (sym1['y'] - sym2['y'])**2)
                if dist < distance_threshold:
                    # Get edge points for both symbols
                    edges1 = self.get_edge_points(sym1['x'], sym1['y'], sym1['image'], sym1['angle'])
                    edges2 = self.get_edge_points(sym2['x'], sym2['y'], sym2['image'], sym2['angle'])
                    
                    # Find closest edge points
                    min_dist = float('inf')
                    best_edge1 = None
                    best_edge2 = None
                    
                    for e1 in edges1:
                        for e2 in edges2:
                            d = math.sqrt((e1[0] - e2[0])**2 + (e1[1] - e2[1])**2)
                            if d < min_dist:
                                min_dist = d
                                best_edge1 = e1
                                best_edge2 = e2
                    
                    if best_edge1 and best_edge2:
                        potential_connections.append({
                            'i': i,
                            'j': j,
                            'dist': dist,
                            'edge1': best_edge1,
                            'edge2': best_edge2
                        })
        
        # Sort by distance (closer connections first)
        potential_connections.sort(key=lambda x: x['dist'])
        
        # Phase 1: Ensure each symbol has at least 1 connection
        for conn in potential_connections:
            i, j = conn['i'], conn['j']
            if connection_count[i] == 0 or connection_count[j] == 0:
                # Make this connection
                connections_made.append((i, j))
                connection_count[i] += 1
                connection_count[j] += 1
        
        # Phase 2: Add more connections up to max 3 per symbol
        for conn in potential_connections:
            i, j = conn['i'], conn['j']
            # Skip if already connected
            if (i, j) in connections_made:
                continue
            # Skip if either symbol already has 3 connections
            if connection_count[i] >= 3 or connection_count[j] >= 3:
                continue
            # Random chance to add connection
            if random.random() < connection_prob:
                connections_made.append((i, j))
                connection_count[i] += 1
                connection_count[j] += 1
        
        # Draw all connections
        for conn in potential_connections:
            i, j = conn['i'], conn['j']
            if (i, j) in connections_made:
                edge1 = conn['edge1']
                edge2 = conn['edge2']
                
                # Random line width
                line_width = random.randint(1, 2)
                # Random line color (dark gray to black)
                line_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
                
                # Randomly choose between straight line and L-shaped connection
                use_l_shape = random.random() < 0.5
                
                if use_l_shape:
                    # Draw L-shaped connection
                    result = self.draw_l_shaped_connection(
                        result,
                        (int(edge1[0]), int(edge1[1])),
                        (int(edge2[0]), int(edge2[1])),
                        line_color,
                        line_width
                    )
                else:
                    # Draw straight line
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
            
            # Ensure coordinates are within image bounds
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


def generate_realistic_dataset(output_dir: str = 'output', num_images: int = 30,
                              symbols_dir: str = 'Instruments', width: int = 1920, height: int = 1080):
    """Generate realistic schematic diagram dataset."""
    
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
    print(f"Generating {num_images} realistic schematic diagrams...")
    
    for i in range(num_images):
        # Create white background
        background = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Prepare symbols for this image
        symbols_for_image = []
        num_symbols = random.randint(5, 20)
        for _ in range(num_symbols):
            sym_img, sym_label = symbol_loader.get_random_symbol()
            symbols_for_image.append((sym_img, sym_label))
        
        # Place symbols with spatial transforms
        canvas, placed_symbols = composer.place_symbols(
            background, symbols_for_image, spatial_processor
        )
        
        # Connect symbols edge-to-edge with wires/lines
        canvas = composer.connect_symbols_edge_to_edge(
            canvas, placed_symbols,
            connection_prob=random.uniform(0.3, 0.5),
            distance_threshold=random.randint(400, 600)
        )
        
        # Apply pixel-level transforms to entire image
        canvas = pixel_processor.apply_pixel_transforms(canvas)
        
        # Extract polygon coordinates
        annotations = composer.extract_polygon_coordinates(placed_symbols)
        
        # Save image
        image_filename = f"schematic_{i+1:03d}.png"
        image_path = images_dir / image_filename
        cv2.imwrite(str(image_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        
        # Save annotation
        annotation_filename = f"schematic_{i+1:03d}.txt"
        annotation_path = annotations_dir / annotation_filename
        annotation_writer.write_annotation(str(annotation_path), annotations)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i + 1}/{num_images} images")
    
    print(f"\nDataset generation complete!")
    print(f"Images saved to: {images_dir}")
    print(f"Annotations saved to: {annotations_dir}")


if __name__ == '__main__':
    generate_realistic_dataset(
        output_dir='output',
        num_images=100,
        symbols_dir='Instruments',
        width=1920,
        height=1080
    )
