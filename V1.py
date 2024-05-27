import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2
import random
import copy


# Function to convert image to grayscale
def convert_to_grayscale(image):
    return ImageOps.grayscale(image)


# Function to calculate the difference between two images
def calculate_difference(image1, image2):
    diff = np.sum(np.abs(np.array(image1) - np.array(image2)))
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


# Function to draw a shape on an image
def draw_shape(image, shape):
    draw = ImageDraw.Draw(image)
    if shape['type'] == 'ellipse':
        draw.ellipse(shape['coords'], fill=shape['color'])
    elif shape['type'] == 'rectangle':
        draw.rectangle(shape['coords'], fill=shape['color'])
    return image


# Genetic Algorithm to evolve the image
def evolve_image(target_image, num_generations, num_shapes, keep_scale=True, grayscale=False):
    target_array = np.array(target_image)
    width, height = target_image.size

    # Initialize with a blank image
    best_image = Image.new('RGB', (width, height), (255, 255, 255))
    best_diff = calculate_difference(target_image, best_image)

    for generation in range(num_generations):
        new_image = copy.deepcopy(best_image)
        for _ in range(num_shapes):
            shape = create_random_shape((width, height), keep_scale)
            new_image = draw_shape(new_image, shape)

        if grayscale:
            new_image = convert_to_grayscale(new_image).convert('RGB')

        new_diff = calculate_difference(target_image, new_image)

        if new_diff < best_diff:
            best_image = new_image
            best_diff = new_diff
            print(f"Generation {generation + 1}: Improved difference to {best_diff}")
        else:
            print(f"Generation {generation + 1}: No improvement")

    return best_image


# Main function to run the algorithm
def main():
    target_image_path = './002.png'
    target_image = Image.open(target_image_path)

    num_generations = 1000
    num_shapes_per_generation = 5
    keep_scale = True
    grayscale = False

    evolved_image = evolve_image(target_image, num_generations, num_shapes_per_generation, keep_scale, grayscale)
    evolved_image.save('evolved_image.jpg')


if __name__ == "__main__":
    main()
