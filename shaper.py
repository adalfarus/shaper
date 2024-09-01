from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy
from aplustools.security.rand import WeightedRandom
from concurrent.futures import ThreadPoolExecutor
from aplustools.package.timid import TimidTimer
from PySide6.QtCore import Qt, QThread, Signal
from PIL import Image, ImageQt, ImageDraw
from PySide6.QtGui import QPixmap
from typing import Optional
import numpy as np
import time
import cv2
import sys
import os

# Random number generator
strong_rng = WeightedRandom()


class LODer:
    def __init__(self, x: int = 2500, y: int = 3500):
        self.scaler = min(x, y) / 2500
        self.weighted_list = self._create_weighted_list()
        self.lods = [self.lod1, self.lod2, self.lod3, self.lod4, self.lod5, self.lod6]
        self.current_lod_index = 0

    def _create_weighted_list(self):
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
        return weighted_list

    def _weighted_random(self, pairs):
        total = sum(pair[0] for pair in pairs)
        r = strong_rng.randint(1, total)
        for (weight, value) in pairs:
            r -= weight
            if r <= 0:
                return value

    def lod1(self):
        scale_factor = round(strong_rng.exponential_distribution(0.1, 10, 5.0), 1) * 4
        return scale_factor * self.scaler

    def lod2(self):
        scale_factor = self._weighted_random(self.weighted_list) * 4
        return scale_factor * self.scaler

    def lod3(self):
        scale_factor = strong_rng.uniform(0.1, 0.5) * 2.5
        return scale_factor * self.scaler

    def lod4(self):
        scale_factor = strong_rng.uniform(0.1, 0.5)
        return scale_factor * self.scaler

    def lod5(self):
        scale_factor = strong_rng.uniform(0.1, 0.3)
        return scale_factor * self.scaler

    def lod6(self):
        scale_factor = strong_rng.uniform(0.05, 0.2)
        return scale_factor * self.scaler

    def get_current_lod(self):
        return self.lods[self.current_lod_index]()

    def increase_lod(self):
        if self.current_lod_index < len(self.lods) - 1:
            self.current_lod_index += 1

    def reset_lod(self):
        self.current_lod_index = 0


class ImageProcessor:
    @staticmethod
    def calculate_difference(image1, image2):
        diff = np.sum(np.abs(image1.astype("float") - image2.astype("float")))
        return diff

    @staticmethod
    def create_random_shape(image_size, keep_scale=True):
        width, height = image_size
        shape_type = strong_rng.choice(['ellipse', 'rectangle'])
        x1, y1 = strong_rng.randint(0, width), strong_rng.randint(0, height)
        if keep_scale:
            x2, y2 = strong_rng.randint(x1, width), strong_rng.randint(y1, height)
        else:
            x2, y2 = strong_rng.randint(0, width), strong_rng.randint(0, height)
        color = tuple(strong_rng.randint(0, 255) for _ in range(3))
        shape = {'type': shape_type, 'coords': (x1, y1, x2, y2), 'color': color}
        return shape

    @staticmethod
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

    @staticmethod
    def create_random_shape_from_image(shape_image, target_image, x, y, scale_factor, apply_grayscale=False):
        new_shape_size = (int(shape_image.shape[1] * scale_factor), int(shape_image.shape[0] * scale_factor))
        shape_image = cv2.resize(shape_image, new_shape_size)

        angle = strong_rng.uniform(0, 360)
        center = (shape_image.shape[1] // 2, shape_image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        shape_image = cv2.warpAffine(shape_image, matrix, (shape_image.shape[1], shape_image.shape[0]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        if apply_grayscale:
            alpha_channel = shape_image[:, :, 3] / 255.0
            non_transparent_indices = alpha_channel > 0
            shape_image_gray = cv2.cvtColor(shape_image, cv2.COLOR_BGR2GRAY)
            shape_image_gray = cv2.cvtColor(shape_image_gray, cv2.COLOR_GRAY2RGBA)
            sample_x = min(x, target_image.shape[1] - new_shape_size[0])
            sample_y = min(y, target_image.shape[0] - new_shape_size[1])
            sampled_color = target_image[sample_y:sample_y + new_shape_size[1], sample_x:sample_x + new_shape_size[0]].mean(axis=(0, 1))
            for c in range(3):
                shape_image_gray[:, :, c] = shape_image_gray[:, :, c] * (sampled_color[c] / 255.0)
            shape_image[non_transparent_indices] = shape_image_gray[non_transparent_indices]

        return shape_image, angle

    @staticmethod
    def blend_shape(image, shape, x, y, alpha, angle):
        height, width = shape.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_shape = cv2.warpAffine(shape, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        blended = image.copy()
        alpha_channel = rotated_shape[:, :, 3] / 255.0
        for c in range(3):
            blended[y:y + height, x:x + width, c] = (1 - alpha * alpha_channel) * blended[y:y + height, x:x + width, c] + alpha * alpha_channel * rotated_shape[:, :, c]
        return blended

    @staticmethod
    def calculate_improvement(target_image, current_image, shape, x, y, alpha, angle):
        height, width = shape.shape[:2]
        if x + width > target_image.shape[1] or y + height > target_image.shape[0]:
            return None  # Skip shapes that go out of bounds

        blended_image = ImageProcessor.blend_shape(current_image, shape, x, y, alpha, angle)
        new_diff = ImageProcessor.calculate_difference(target_image, blended_image)
        current_diff = ImageProcessor.calculate_difference(target_image, current_image)

        if new_diff < current_diff:
            return blended_image, new_diff, shape, x, y, alpha, angle
        else:
            return None


class ShapeAdder(QThread):
    image_updated = Signal(np.ndarray)

    def __init__(self, target_image, image_shapes, shape_images_paths, use_random_shapes=False, apply_grayscale=False,
                 old: Optional[str] = None):
        super().__init__()
        self.original_target_image = target_image
        self.target_image = cv2.GaussianBlur(target_image, (25, 25), 0)  # Apply Gaussian blur to the target image (55)
        self.image_shapes: list[np.ndarray] = image_shapes
        self.shape_images_paths: list[str] = shape_images_paths
        self.use_random_shapes = use_random_shapes
        self.apply_grayscale = apply_grayscale
        self.target_array = target_image.astype("float")
        self.current_image = cv2.cvtColor(cv2.imread(old), cv2.COLOR_BGR2RGB) if old else np.ones_like(self.target_array) * 255  # Start with a blank
        self.executor = ThreadPoolExecutor(max_workers=4)  # Thread pool with 4 workers
        self.loder = LODer(*target_image.shape[0:2])
        self.no_improvement_count = 0
        self._running = False
        self._stopped = True
        self.image_shapes_count = [0] * len(image_shapes)
        self.current_diff = 0
        self.base_diff = ImageProcessor.calculate_difference(target_image, np.ones_like(self.target_array) * 255)

    def run(self):
        self._running = True
        self._stopped = False
        while self._running:
            x = strong_rng.randint(0, self.target_image.shape[1] - 1)
            y = strong_rng.randint(0, self.target_image.shape[0] - 1)
            alpha = strong_rng.uniform(0.1, 1.0)

            shape_image_index = None
            if self.use_random_shapes:
                shape = ImageProcessor.create_random_shape((self.target_image.shape[1], self.target_image.shape[0]))
                future = self.executor.submit(self.add_random_shape, self.current_image, shape, alpha)
            else:
                shape_image_index = strong_rng.randint(0, len(self.image_shapes) - 1)
                shape, angle = ImageProcessor.create_random_shape_from_image(
                    self.image_shapes[shape_image_index], self.target_image, x, y, self.loder.get_current_lod(), self.apply_grayscale)
                future = self.executor.submit(ImageProcessor.calculate_improvement, self.target_image,
                                              self.current_image, shape, x, y, alpha, angle)

            result = future.result()
            if result:
                if shape_image_index is not None:
                    self.image_shapes_count[shape_image_index] += 1
                self.current_image, self.current_diff, shape, x, y, alpha, angle = result
                self.image_updated.emit(self.current_image)  # (self.current_image)
                self.no_improvement_count = 0  # Reset counter on improvement
            else:
                self.no_improvement_count += 1
                if self.no_improvement_count > 40 and 5 > self.loder.current_lod_index >= 3:  # basically at max level
                    print(f"Increasing detail, lod {self.loder.current_lod_index + 1}")
                    self.loder.increase_lod()
                    self.no_improvement_count = 0
                    blur_target = {4: 15, 5: 11, 6: 9, 7: 5}[self.loder.current_lod_index]
                    self.target_image = cv2.GaussianBlur(self.original_target_image, (blur_target, blur_target), 0)
                elif self.no_improvement_count > 10 and self.loder.current_lod_index < 3:  # Change level of detail after 10 failed attempts
                    print(f"lod {self.loder.current_lod_index + 1}")
                    # if self.loder.current_lod_index == 0:
                    #     self.target_image = cv2.GaussianBlur(self.target_image, (25, 25), 0)
                    self.loder.increase_lod()
                    self.no_improvement_count = 0
        self._stopped = True

    def stop(self):
        self._running = False
        while not self._stopped:
            time.sleep(0.01)

    def add_random_shape(self, image, shape, alpha):
        new_image = ImageProcessor.draw_shape(image, shape, alpha)
        new_diff = ImageProcessor.calculate_difference(self.target_array, new_image)
        current_diff = ImageProcessor.calculate_difference(self.target_array, image)
        if new_diff < current_diff:
            return new_image, new_diff
        return None


class ImageWindow(QMainWindow):
    def __init__(self, shape_adder: ShapeAdder):
        super().__init__()
        self.shape_adder = shape_adder

        # Get monitor size and calculate new window size
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        max_width = screen_geometry.width() // 2
        max_height = screen_geometry.height() // 2
        self.original_height, self.original_width = shape_adder.target_image.shape[:2]

        while True:
            if self.original_height > max_height:
                self.original_width, self.original_height = self.scale_original_size_to(height=max_height)
            elif self.original_width > max_width:
                self.original_width, self.original_height = self.scale_original_size_to(width=max_width)
            else:
                break

        self.resize(self.original_width, self.original_height)  # Resize to half of the monitors size in both dirs

        self.last_resize_time = time.time()
        self.wanted_size = self.size()
        self.t = TimidTimer(start_now=False)
        self.t.warmup_fire()
        self.t.fire(0.1, self.update_size)

        self.initUI()
        self.update_image(self.shape_adder.current_image)
        self.shape_adder.image_updated.connect(self.update_image)
        self.shape_adder.start()
        self.current_image = None

    def scale_original_size_to(self, width: Optional[int] = None, height: Optional[int] = None,
                               keep_aspect_ration: bool = False):
        if width:
            aspect_ratio = self.original_height / self.original_width
            return width, int(width * aspect_ratio)
        elif height:
            aspect_ratio = self.original_width / self.original_height
            return int(height * aspect_ratio), height
        elif width and height:
            if keep_aspect_ration:
                if width < height:
                    return self.scale_original_size_to(width=width)
                else:
                    return self.scale_original_size_to(height=height)
            else:
                return width, height

    def update_size(self):
        if time.time() - self.last_resize_time > 0.1:  # and not self.isMaximized():
            self.wanted_size = self.image_label.pixmap().size()

    def initUI(self):
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.image_label)
        # self.showMaximized()
        self.show()

    def resizeEvent(self, event):
        if event.size().width() > 100 and event.size().width() > 100:
            self.wanted_size = event.size()
            if hasattr(self, 'current_image'):
                self.update_image(self.current_image)
            self.last_resize_time = time.time()
            event.accept()
        else:
            event.ignore()

    def update_image(self, image):
        if image is None:
            return
        if self.wanted_size != self.size() and self.wanted_size.width() > 100 and self.wanted_size.height() > 100:
            self.resize(self.wanted_size)

        self.setWindowTitle(str(round(100 - ((self.shape_adder.current_diff / self.shape_adder.base_diff) * 100), 2) % 100) + "%")
        self.current_image = image
        qt_image = ImageQt.ImageQt(Image.fromarray(image.astype('uint8')))
        if self.width() > self.height():
            pixmap = QPixmap.fromImage(qt_image).scaledToHeight(self.height(), Qt.FastTransformation)  # Qt.IgnoreAspectRatio,
        else:
            pixmap = QPixmap.fromImage(qt_image).scaledToWidth(self.width(), Qt.FastTransformation)  # Qt.IgnoreAspectRatio,
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if hasattr(self, 'current_image'):
            final_image = Image.fromarray(self.current_image.astype('uint8'))
            final_image.save('final_image.png')
        print({os.path.basename(k): v for k, v in zip(self.shape_adder.shape_images_paths, self.shape_adder.image_shapes_count)})
        self.shape_adder.stop()
        self.shape_adder.quit()
        self.t.stop_fire()
        event.accept()


# Main function to run the algorithm
def main(target_image_path, shapes_dir, change_color, use_random_shapes, old):
    target_image = cv2.imread(target_image_path)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    shape_images_paths = [os.path.join(shapes_dir, x) for x in os.listdir(shapes_dir)]
    shape_images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in shape_images_paths]

    apply_grayscale = change_color  # Convert shapes to grayscale and recolor
    use_random_shapes = use_random_shapes  # Set this to True to use the new random shapes instead of image-based shapes

    shape_adder = ShapeAdder(target_image, shape_images, shape_images_paths, use_random_shapes, apply_grayscale, old)

    app = QApplication(sys.argv)
    window = ImageWindow(shape_adder)
    sys.exit(app.exec())


if __name__ == "__main__":
    if len(sys.argv) < 5:
        main('002.png', "./shapes", True, False, None)
    else:
        main(*sys.argv[1:3], sys.argv[3] == "True", sys.argv[4] == "True", None if len(sys.argv) < 6 else sys.argv[5])
