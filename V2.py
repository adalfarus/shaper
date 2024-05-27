import numpy as np
import cv2
import random
from PIL import Image, ImageQt, ImageDraw
from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QMainWindow, QSizePolicy
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from random import randint

def weighted_random(pairs):
    total = sum(pair[0] for pair in pairs)
    r = randint(1, total)
    for (weight, value) in pairs:
        r -= weight
        if r <= 0:
            return value

# Function to calculate the difference between two images
def calculate_difference(image1, image2):
    diff = np.sum(np.abs(image1.astype("float") - image2.astype("float")))
    return diff

# Function to create random shapes
def create_random_shape(image_size, keep_scale=True):
    width, height = image_size
    shape_type = random.choice(['ellipse', 'rectangle'])
    x1, y1 = random.randint(0, width), random.randint(0, height)
    if keep_scale:
        x2, y2 = random.randint(x1, width), random.randint(y1, height)
    else:
        x2, y2 = random.randint(0, width), random.randint(0, height)
    color = tuple(random.randint(0, 255) for _ in range(3))
    shape = {'type': shape_type, 'coords': (x1, y1, x2, y2), 'color': color}
    return shape

# Function to draw a random shape on an image
def draw_shape(image, shape, alpha):
    overlay = image.copy()
    output = image.copy()
    draw = ImageDraw.Draw(overlay)
    if shape['type'] == 'ellipse':
        draw.ellipse(shape['coords'], fill=shape['color'])
    elif shape['type'] == 'rectangle':
        draw.rectangle(shape['coords'], fill=shape['color'])
    blended = Image.blend(output, overlay, alpha)
    return np.array(blended)

# Random size
weighted_list = []
remaining = 100  # start with the full value of 100
weight = 0.6  # 60%
increment = 0.1  # weight increment

current_weight = 0.0

while remaining > 1:  # stop when the remaining value is less than 1
    current_weight += increment
    value = int(remaining * weight)
    weighted_list.append((value, round(current_weight, 1)))
    remaining -= value

# Add the last remaining part if it's still greater than 0
if remaining > 0:
    current_weight += increment
    weighted_list.append((remaining, round(current_weight, 1)))

# Function to create a random shape from a set of images with rotation, scaling, and opacity
def create_random_shape_from_image(image_shapes, target_image, x, y, apply_grayscale=False):
    shape_image = random.choice(image_shapes)

    scale_factor = weighted_random(weighted_list) * 4  # lod 3 (very low) (Contributuion 100%)
    scale_factor = random.uniform(0.1, 0.5)  # lod 1 (high) (Contribution 25%) # * 2.5 lod 2 (low) (Contribution 50%)
    new_shape_size = (int(shape_image.shape[1] * scale_factor), int(shape_image.shape[0] * scale_factor))
    shape_image = cv2.resize(shape_image, new_shape_size)

    # Random rotation
    angle = random.uniform(0, 360)
    center = (shape_image.shape[1] // 2, shape_image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    shape_image = cv2.warpAffine(shape_image, matrix, (shape_image.shape[1], shape_image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    # Apply grayscale and color sampling only to non-transparent parts
    if apply_grayscale:
        alpha_channel = shape_image[:, :, 3] / 255.0
        non_transparent_indices = alpha_channel > 0
        shape_image_gray = cv2.cvtColor(shape_image, cv2.COLOR_BGR2GRAY)
        shape_image_gray = cv2.cvtColor(shape_image_gray, cv2.COLOR_GRAY2RGBA)
        sample_x = min(x, target_image.shape[1] - new_shape_size[0])
        sample_y = min(y, target_image.shape[0] - new_shape_size[1])
        sampled_color = target_image[sample_y:sample_y+new_shape_size[1], sample_x:sample_x+new_shape_size[0]].mean(axis=(0, 1))
        for c in range(3):
            shape_image_gray[:, :, c] = shape_image_gray[:, :, c] * (sampled_color[c] / 255.0)
        shape_image[non_transparent_indices] = shape_image_gray[non_transparent_indices]

    return shape_image, angle

# Function to blend a shape onto an image
def blend_shape(image, shape, x, y, alpha, angle):
    height, width = shape.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_shape = cv2.warpAffine(shape, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    blended = image.copy()
    alpha_channel = rotated_shape[:, :, 3] / 255.0
    for c in range(3):
        blended[y:y+height, x:x+width, c] = (1 - alpha * alpha_channel) * blended[y:y+height, x:x+width, c] + alpha * alpha_channel * rotated_shape[:, :, c]
    return blended

# Function to calculate if adding a shape improves the image
def calculate_improvement(target_image, current_image, shape, x, y, alpha, angle):
    height, width = shape.shape[:2]
    if x + width > target_image.shape[1] or y + height > target_image.shape[0]:
        return None  # Skip shapes that go out of bounds

    blended_image = blend_shape(current_image, shape, x, y, alpha, angle)
    new_diff = calculate_difference(target_image, blended_image)
    current_diff = calculate_difference(target_image, current_image)

    if new_diff < current_diff:
        return blended_image, new_diff, shape, x, y, alpha, angle
    else:
        return None

# Continuous Shape Addition

# Continuous Shape Addition
class ShapeAdder(QThread):
    image_updated = Signal(np.ndarray)

    def __init__(self, target_image, image_shapes, use_random_shapes=False, apply_grayscale=False):
        super().__init__()
        self.target_image = target_image
        self.image_shapes = image_shapes
        self.use_random_shapes = use_random_shapes
        self.apply_grayscale = apply_grayscale
        self.target_array = target_image.astype("float")
        self.current_image = np.ones_like(self.target_array) * 255  # Start with a white image
        self.executor = ThreadPoolExecutor(max_workers=4)  # Thread pool with 4 workers

    def run(self):
        while True:
            x = random.randint(0, self.target_image.shape[1] - 1)
            y = random.randint(0, self.target_image.shape[0] - 1)
            if self.use_random_shapes:
                shape = create_random_shape((self.target_image.shape[1], self.target_image.shape[0]))
                alpha = random.uniform(0.1, 1.0)
                future = self.executor.submit(self.add_random_shape, self.current_image, shape, alpha)
            else:
                shape, angle = create_random_shape_from_image(self.image_shapes, self.target_image, x, y,
                                                              self.apply_grayscale)
                alpha = random.uniform(0.1, 1.0)
                future = self.executor.submit(calculate_improvement, self.target_image, self.current_image, shape, x, y,
                                              alpha, angle)

            result = future.result()
            if result:
                self.current_image, _, shape, x, y, alpha, angle = result
                self.image_updated.emit(self.current_image)

    def add_random_shape(self, image, shape, alpha):
        print("H")
        new_image = draw_shape(image, shape, alpha)
        print("H")
        new_diff = calculate_difference(self.target_array, new_image)
        print("H")
        current_diff = calculate_difference(self.target_array, image)
        print("H")
        if new_diff < current_diff:
            return new_image, new_diff
        return None

class ImageWindow(QMainWindow):
    def __init__(self, shape_adder):
        super().__init__()
        self.shape_adder = shape_adder
        self.initUI()
        self.shape_adder.image_updated.connect(self.update_image)
        self.shape_adder.start()

    def initUI(self):
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)
        self.showMaximized()

    def resizeEvent(self, event):
        if hasattr(self, 'current_image'):
            self.update_image(self.current_image)

    def update_image(self, image):
        self.current_image = image
        qt_image = ImageQt.ImageQt(Image.fromarray(image.astype('uint8')))
        pixmap = QPixmap.fromImage(qt_image).scaled(self.size(), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if hasattr(self, 'current_image'):
            final_image = Image.fromarray(self.current_image.astype('uint8'))
            final_image.save('final_image.png')
        event.accept()


# Main function to run the algorithm
def main():
    target_image_path = '002.png'
    target_image = cv2.imread(target_image_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    shape_images_paths = [os.path.join("./shapes", x) for x in os.listdir("./shapes")]
    shape_images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in shape_images_paths]

    num_generations = 99999
    population_size = 10
    apply_grayscale = True  # Convert shapes to grayscale and recolor
    use_random_shapes = False  # Set this to True to use the new random shapes instead of image-based shapes

    shape_adder = ShapeAdder(target_image, shape_images, use_random_shapes, apply_grayscale)
    # evolver.initialize_population()

    app = QApplication(sys.argv)
    window = ImageWindow(shape_adder)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
