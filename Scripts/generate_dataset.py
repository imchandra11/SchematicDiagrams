import cv2
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict
import albumentations as A


class BackgroundGenerator:
    """Generate diverse synthetic backgrounds for schematic diagrams."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
    
    def generate_gradient(self) -> np.ndarray:
        """Generate gradient backgrounds (linear, radial, diagonal, multi-directional)."""
        bg_type = random.choice(['linear_h', 'linear_v', 'linear_d', 'radial', 
                                'radial_multi', 'corner_radial', 'vertical_split', 'horizontal_split'])
        
        if bg_type == 'linear_h':
            # Horizontal gradient
            start = random.randint(180, 240)
            end = random.randint(start, 255)
            bg = np.linspace(start, end, self.width, dtype=np.uint8)
            bg = np.tile(bg, (self.height, 1))
            bg = cv2.merge([bg, bg, bg])
        
        elif bg_type == 'linear_v':
            # Vertical gradient
            start = random.randint(180, 240)
            end = random.randint(start, 255)
            bg = np.linspace(start, end, self.height, dtype=np.uint8)
            bg = np.tile(bg, (self.width, 1)).T
            bg = cv2.merge([bg, bg, bg])
        
        elif bg_type == 'linear_d':
            # Diagonal gradient
            start = random.randint(180, 240)
            end = random.randint(start, 255)
            bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for i in range(self.height):
                for j in range(self.width):
                    intensity = int(start + (end - start) * (i + j) / (self.height + self.width))
                    bg[i, j] = [intensity, intensity, intensity]
        
        elif bg_type == 'radial':
            # Radial gradient from center
            center_x, center_y = self.width // 2, self.height // 2
            max_dist = math.sqrt(center_x**2 + center_y**2)
            center_intensity = random.randint(180, 240)
            edge_intensity = random.randint(center_intensity, 255)
            bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for i in range(self.height):
                for j in range(self.width):
                    dist = math.sqrt((i - center_y)**2 + (j - center_x)**2)
                    intensity = int(center_intensity + (edge_intensity - center_intensity) * (1 - dist / max_dist))
                    bg[i, j] = [intensity, intensity, intensity]
        
        elif bg_type == 'radial_multi':
            # Multiple radial gradients
            bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * random.randint(220, 255)
            num_centers = random.randint(2, 4)
            for _ in range(num_centers):
                cx = random.randint(0, self.width)
                cy = random.randint(0, self.height)
                max_dist = random.randint(300, 600)
                center_int = random.randint(190, 240)
                edge_int = random.randint(center_int, 255)
                for i in range(0, self.height, 2):
                    for j in range(0, self.width, 2):
                        dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                        if dist < max_dist:
                            intensity = int(center_int + (edge_int - center_int) * (1 - dist / max_dist))
                            intensity = max(0, min(255, intensity))
                            bg[i, j] = [intensity, intensity, intensity]
        
        elif bg_type == 'corner_radial':
            # Radial gradient from corner
            corner = random.choice(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
            if corner == 'top_left':
                cx, cy = 0, 0
            elif corner == 'top_right':
                cx, cy = self.width, 0
            elif corner == 'bottom_left':
                cx, cy = 0, self.height
            else:  # bottom_right
                cx, cy = self.width, self.height
            max_dist = math.sqrt(self.width**2 + self.height**2)
            center_intensity = random.randint(180, 240)
            edge_intensity = random.randint(center_intensity, 255)
            bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for i in range(self.height):
                for j in range(self.width):
                    dist = math.sqrt((i - cy)**2 + (j - cx)**2)
                    intensity = int(center_intensity + (edge_intensity - center_intensity) * (dist / max_dist))
                    bg[i, j] = [intensity, intensity, intensity]
        
        elif bg_type == 'vertical_split':
            # Vertical split gradient
            split_x = random.randint(self.width // 4, 3 * self.width // 4)
            left_int = random.randint(180, 240)
            right_int = random.randint(180, 240)
            bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for j in range(self.width):
                if j < split_x:
                    intensity = int(left_int + (255 - left_int) * j / split_x)
                else:
                    intensity = int(right_int + (255 - right_int) * (j - split_x) / (self.width - split_x))
                bg[:, j] = [intensity, intensity, intensity]
        
        else:  # horizontal_split
            # Horizontal split gradient
            split_y = random.randint(self.height // 4, 3 * self.height // 4)
            top_int = random.randint(180, 240)
            bottom_int = random.randint(180, 240)
            bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for i in range(self.height):
                if i < split_y:
                    intensity = int(top_int + (255 - top_int) * i / split_y)
                else:
                    intensity = int(bottom_int + (255 - bottom_int) * (i - split_y) / (self.height - split_y))
                bg[i, :] = [intensity, intensity, intensity]
        
        return bg
    
    def generate_texture(self) -> np.ndarray:
        """Generate texture backgrounds (noise patterns, grid patterns, dirty textures)."""
        texture_type = random.choice(['noise', 'grid', 'gaussian_noise', 'heavy_noise', 
                                     'speckle', 'dirty', 'scratched', 'stained'])
        
        if texture_type == 'noise':
            # Uniform noise
            bg = np.random.randint(220, 255, (self.height, self.width, 3), dtype=np.uint8)
        
        elif texture_type == 'gaussian_noise':
            # Gaussian noise
            bg = np.random.normal(240, 10, (self.height, self.width, 3)).astype(np.uint8)
            bg = np.clip(bg, 200, 255)
        
        elif texture_type == 'heavy_noise':
            # Heavy noise with darker variations
            base = np.random.randint(180, 255, (self.height, self.width, 3), dtype=np.uint8)
            noise = np.random.randint(-30, 30, (self.height, self.width, 3), dtype=np.int16)
            bg = np.clip(base.astype(np.int16) + noise, 150, 255).astype(np.uint8)
        
        elif texture_type == 'speckle':
            # Speckle noise pattern
            bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * random.randint(230, 255)
            num_specks = random.randint(500, 2000)
            for _ in range(num_specks):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                intensity = random.randint(150, 240)
                size = random.randint(1, 3)
                cv2.circle(bg, (x, y), size, (intensity, intensity, intensity), -1)
        
        elif texture_type == 'dirty':
            # Dirty/aged paper effect - more dirty
            base_intensity = random.randint(180, 240)
            bg = np.random.randint(base_intensity - 20, base_intensity + 20, (self.height, self.width, 3), dtype=np.uint8)
            bg = np.clip(bg, 150, 255)
            
            # Add many darker patches (more dirt)
            num_patches = random.randint(30, 60)
            for _ in range(num_patches):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                w_patch = random.randint(40, 250)
                h_patch = random.randint(40, 250)
                intensity = random.randint(150, base_intensity - 20)
                cv2.ellipse(bg, (x, y), (w_patch, h_patch), random.randint(0, 360), 0, 360,
                           (intensity, intensity, intensity), -1)
            
            # Add noise overlay
            noise = np.random.randint(-15, 15, (self.height, self.width, 3), dtype=np.int16)
            bg = np.clip(bg.astype(np.int16) + noise, 140, 255).astype(np.uint8)
        
        elif texture_type == 'scratched':
            # Scratched surface
            bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * random.randint(220, 255)
            num_scratches = random.randint(20, 50)
            for _ in range(num_scratches):
                x1 = random.randint(0, self.width)
                y1 = random.randint(0, self.height)
                x2 = random.randint(0, self.width)
                y2 = random.randint(0, self.height)
                intensity = random.randint(190, 230)
                thickness = random.randint(1, 3)
                cv2.line(bg, (x1, y1), (x2, y2), (intensity, intensity, intensity), thickness)
        
        elif texture_type == 'stained':
            # Stained background
            bg = np.random.randint(210, 255, (self.height, self.width, 3), dtype=np.uint8)
            num_stains = random.randint(5, 15)
            for _ in range(num_stains):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                radius = random.randint(30, 150)
                intensity = random.randint(170, 220)
                cv2.circle(bg, (x, y), radius, (intensity, intensity, intensity), -1)
        
        else:  # grid
            # Grid pattern
            bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
            grid_size = random.randint(50, 150)
            line_color = random.randint(230, 250)
            for i in range(0, self.height, grid_size):
                cv2.line(bg, (0, i), (self.width, i), (line_color, line_color, line_color), 1)
            for j in range(0, self.width, grid_size):
                cv2.line(bg, (j, 0), (j, self.height), (line_color, line_color, line_color), 1)
        
        return bg
    
    def generate_solid(self) -> np.ndarray:
        """Generate solid color backgrounds."""
        intensity = random.randint(240, 255)
        bg = np.ones((self.height, self.width, 3), dtype=np.uint8) * intensity
        return bg
    
    def generate_composite(self) -> np.ndarray:
        """Generate composite backgrounds (gradient + texture)."""
        base = self.generate_gradient()
        texture = self.generate_texture()
        alpha = random.uniform(0.3, 0.7)
        bg = cv2.addWeighted(base, 1 - alpha, texture, alpha, 0)
        return bg
    
    def generate(self) -> np.ndarray:
        """Generate a random synthetic background."""
        bg_type = random.choice(['gradient', 'texture', 'solid', 'composite'])
        
        if bg_type == 'gradient':
            return self.generate_gradient()
        elif bg_type == 'texture':
            return self.generate_texture()
        elif bg_type == 'solid':
            return self.generate_solid()
        else:
            return self.generate_composite()


class SymbolProcessor:
    """Process and augment instrument symbols."""
    
    def __init__(self, symbols_dir: str):
        self.symbols_dir = symbols_dir
        self.symbol_paths = list(Path(symbols_dir).glob('*.png'))
        if not self.symbol_paths:
            raise ValueError(f"No PNG files found in {symbols_dir}")
        
        # Albumentations augmentation pipeline
        self.augmentation = A.Compose([
            A.CropAndPad(percent=0.1, keep_size=False, p=0.8),
            A.RandomCropFromBorders(
                crop_left=0.15, crop_right=0.15,
                crop_top=0.15, crop_bottom=0.15, p=0.8
            ),
            A.RandomRotate90(p=0.3),
            A.Blur(blur_limit=3, p=0.6),
            A.GaussNoise(p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.2, p=0.5
            ),
        ])
    
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
    
    def augment_symbol(self, img: np.ndarray) -> np.ndarray:
        """Apply augmentation to symbol image."""
        augmented = self.augmentation(image=img)['image']
        return augmented
    
    def resize_symbol(self, img: np.ndarray, min_size: int = 50, max_size: int = 150) -> np.ndarray:
        """Resize symbol to random size within range while maintaining aspect ratio."""
        h, w = img.shape[:2]
        target_size = random.randint(min_size, max_size)
        
        # Maintain aspect ratio
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized
    
    def get_random_symbol(self) -> Tuple[np.ndarray, str]:
        """Get a random symbol with augmentation and resizing."""
        symbol_path = random.choice(self.symbol_paths)
        img, label = self.load_symbol(symbol_path)
        img = self.augment_symbol(img)
        img = self.resize_symbol(img)
        return img, label


class SchematicComposer:
    """Compose schematic diagrams by placing symbols and connecting them."""
    
    def __init__(self, width: int = 1920, height: int = 1080):
        self.width = width
        self.height = height
    
    def get_bounding_box(self, x: int, y: int, img: np.ndarray, angle: float = 0) -> List[Tuple[int, int]]:
        """Get 4-point bounding box polygon coordinates for a placed symbol.
        
        Returns corners in order: top-left, top-right, bottom-right, bottom-left
        """
        h, w = img.shape[:2]
        
        # Corner points relative to center (order: top-left, top-right, bottom-right, bottom-left)
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
        
        # Convert to list of tuples
        corners_list = [(int(c[0]), int(c[1])) for c in corners]
        
        # Sort corners to ensure correct order: top-left, top-right, bottom-right, bottom-left
        # For a rotated rectangle, identify corners by position
        # Sort by y first (top vs bottom), then by x (left vs right)
        sorted_by_y = sorted(corners_list, key=lambda p: (p[1], p[0]))
        top_two = sorted_by_y[:2]  # Top two corners
        bottom_two = sorted_by_y[2:]  # Bottom two corners
        
        # Sort top two by x (left to right)
        top_left = min(top_two, key=lambda p: p[0])
        top_right = max(top_two, key=lambda p: p[0])
        
        # Sort bottom two by x (left to right)
        bottom_left = min(bottom_two, key=lambda p: p[0])
        bottom_right = max(bottom_two, key=lambda p: p[0])
        
        # Return in correct order: top-left, top-right, bottom-right, bottom-left
        polygon = [top_left, top_right, bottom_right, bottom_left]
        return polygon
    
    def check_collision(self, x: int, y: int, img: np.ndarray, 
                       placed_symbols: List[Dict], padding: int = 20) -> bool:
        """Check if placement would cause collision with existing symbols."""
        h, w = img.shape[:2]
        new_rect = (x - w//2 - padding, y - h//2 - padding, 
                   x + w//2 + padding, y + h//2 + padding)
        
        for symbol in placed_symbols:
            sym_x, sym_y = symbol['x'], symbol['y']
            sym_img = symbol['image']
            sym_h, sym_w = sym_img.shape[:2]
            sym_rect = (sym_x - sym_w//2 - padding, sym_y - sym_h//2 - padding,
                       sym_x + sym_w//2 + padding, sym_y + sym_h//2 + padding)
            
            # Check overlap
            if not (new_rect[2] < sym_rect[0] or new_rect[0] > sym_rect[2] or
                   new_rect[3] < sym_rect[1] or new_rect[1] > sym_rect[3]):
                return True
        return False
    
    def place_symbols(self, background: np.ndarray, symbols: List[Tuple[np.ndarray, str]],
                     num_symbols: int = None, max_attempts: int = 100) -> Tuple[np.ndarray, List[Dict]]:
        """Place symbols on background avoiding collisions."""
        if num_symbols is None:
            num_symbols = random.randint(5, 20)
        
        num_symbols = min(num_symbols, len(symbols))
        selected_symbols = random.sample(symbols, num_symbols)
        
        canvas = background.copy()
        placed_symbols = []
        
        for img, label in selected_symbols:
            # Random rotation (small angles for realism)
            angle = random.uniform(-15, 15)
            
            # Calculate rotated image dimensions for collision detection
            h, w = img.shape[:2]
            if angle != 0:
                rad = math.radians(abs(angle))
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                # Dimensions needed to contain rotated image
                rot_w = int(h * sin_a + w * cos_a)
                rot_h = int(h * cos_a + w * sin_a)
            else:
                rot_w, rot_h = w, h
            
            # Random position using rotated dimensions
            attempts = 0
            while attempts < max_attempts:
                x = random.randint(rot_w//2 + 50, self.width - rot_w//2 - 50)
                y = random.randint(rot_h//2 + 50, self.height - rot_h//2 - 50)
                
                # Create a temporary image with rotated dimensions for collision check
                temp_img = np.zeros((rot_h, rot_w, 3), dtype=np.uint8)
                if not self.check_collision(x, y, temp_img, placed_symbols):
                    break
                attempts += 1
            
            if attempts >= max_attempts:
                continue  # Skip this symbol if can't place
            
            # Rotate image if needed (without clipping, using background color)
            if angle != 0:
                # Sample background color at placement location (use average of surrounding area)
                sample_x1 = max(0, x - rot_w // 4)
                sample_x2 = min(self.width, x + rot_w // 4)
                sample_y1 = max(0, y - rot_h // 4)
                sample_y2 = min(self.height, y + rot_h // 4)
                bg_sample = background[sample_y1:sample_y2, sample_x1:sample_x2]
                if bg_sample.size > 0:
                    bg_color = np.mean(bg_sample.reshape(-1, 3), axis=0).astype(np.uint8)
                else:
                    bg_color = background[y, x].astype(np.uint8)
                
                # Create padded image with background color
                padded_img = np.ones((rot_h, rot_w, 3), dtype=np.uint8) * bg_color
                
                # Calculate offset to center the original image in the padded image
                offset_x = (rot_w - w) // 2
                offset_y = (rot_h - h) // 2
                padded_img[offset_y:offset_y + h, offset_x:offset_x + w] = img
                
                # Rotate around center of padded image
                center = (rot_w // 2, rot_h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                img_rotated = cv2.warpAffine(padded_img, rotation_matrix, (rot_w, rot_h),
                                            borderValue=tuple(map(int, bg_color)))
            else:
                img_rotated = img
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
                'image': img,  # Original image for polygon calculation
                'label': label,
                'angle': angle,
                'rotated_image': img_rotated
            })
        
        return canvas, placed_symbols
    
    def connect_symbols(self, canvas: np.ndarray, placed_symbols: List[Dict],
                       connection_prob: float = 0.4, distance_threshold: int = 500) -> np.ndarray:
        """Connect symbols with lines/pipes probabilistically."""
        result = canvas.copy()
        
        for i, sym1 in enumerate(placed_symbols):
            for sym2 in placed_symbols[i+1:]:
                # Calculate distance
                dist = math.sqrt((sym1['x'] - sym2['x'])**2 + (sym1['y'] - sym2['y'])**2)
                
                if dist < distance_threshold and random.random() < connection_prob:
                    # Determine connection style
                    style = random.choice(['straight', 'l_shaped'])
                    
                    # Get connection points (center of symbol edges)
                    x1, y1 = sym1['x'], sym1['y']
                    x2, y2 = sym2['x'], sym2['y']
                    
                    # Line color and width
                    line_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
                    line_width = random.randint(2, 4)
                    
                    if style == 'straight':
                        # Straight line
                        cv2.line(result, (x1, y1), (x2, y2), line_color, line_width)
                    else:
                        # L-shaped connection (horizontal then vertical or vice versa)
                        mid_x = (x1 + x2) // 2
                        mid_y = (y1 + y2) // 2
                        
                        if random.random() < 0.5:
                            # Horizontal then vertical
                            cv2.line(result, (x1, y1), (mid_x, y1), line_color, line_width)
                            cv2.line(result, (mid_x, y1), (mid_x, y2), line_color, line_width)
                            cv2.line(result, (mid_x, y2), (x2, y2), line_color, line_width)
                        else:
                            # Vertical then horizontal
                            cv2.line(result, (x1, y1), (x1, mid_y), line_color, line_width)
                            cv2.line(result, (x1, mid_y), (x2, mid_y), line_color, line_width)
                            cv2.line(result, (x2, mid_y), (x2, y2), line_color, line_width)
        
        return result
    
    def extract_polygon_coordinates(self, placed_symbols: List[Dict]) -> List[Dict]:
        """Extract 4-point polygon coordinates for each placed symbol."""
        annotations = []
        
        for symbol in placed_symbols:
            x, y = symbol['x'], symbol['y']
            img = symbol['image']  # Use original image for polygon calculation
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
        """Write annotation file in CRAFT ICDAR format.
        
        Format: x1,y1,x2,y2,x3,y3,x4,y4,transcription
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for ann in annotations:
                polygon = ann['polygon']
                label = ann['label']
                
                # Format: x1,y1,x2,y2,x3,y3,x4,y4,transcription
                coords = ','.join([f"{p[0]},{p[1]}" for p in polygon])
                line = f"{coords},{label}\n"
                f.write(line)


def generate_dataset(output_dir: str = 'output', num_images: int = 100,
                    symbols_dir: str = 'Instruments', width: int = 1920, height: int = 1080):
    """Generate synthetic schematic diagram dataset."""
    
    # Create output directories
    images_dir = Path(output_dir) / 'images'
    annotations_dir = Path(output_dir) / 'annotations'
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    bg_generator = BackgroundGenerator(width, height)
    symbol_processor = SymbolProcessor(symbols_dir)
    composer = SchematicComposer(width, height)
    annotation_writer = AnnotationWriter()
    
    print(f"Loaded {len(symbol_processor.symbol_paths)} symbols")
    print(f"Generating {num_images} schematic diagrams...")
    
    for i in range(num_images):
        # Generate background
        background = bg_generator.generate()
        
        # Prepare symbols for this image (with augmentation and resizing)
        symbols_for_image = []
        num_symbols = random.randint(5, 20)
        for _ in range(num_symbols):
            sym_img, sym_label = symbol_processor.get_random_symbol()
            symbols_for_image.append((sym_img, sym_label))
        
        # Place symbols
        canvas, placed_symbols = composer.place_symbols(background, symbols_for_image)
        
        # Connect symbols with lines
        canvas = composer.connect_symbols(canvas, placed_symbols,
                                         connection_prob=random.uniform(0.3, 0.5),
                                         distance_threshold=random.randint(400, 600))
        
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
    generate_dataset(
        output_dir='output',
        num_images=30,
        symbols_dir='Instruments',
        width=1920,
        height=1080
    )
