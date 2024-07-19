import cv2
import numpy as np
import os
import json
import random

# define data directories
base_dir = os.path.dirname(os.path.abspath("C:\\Users\\Jordan\\Downloads\\Downloads"))
input_dir = r"C:\Users\Jordan\Downloads\input_data"
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')
label_dir = os.path.join(base_dir, 'labels')

# ensure directories exist
os.makedirs(input_dir, exist_ok=True)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)
print(input_dir)
# minimum size
min_width = 512  # update based on your preferences
min_height = 512

# number of images to be generated from a single input image
num_images = 3

# color mode
color_mode = "rgb"


def apply_distortions(image, distortions):
    # apply psf blur
    if 'psf_blur' in distortions:
        ksize = int(distortions['psf_blur'])
        if ksize % 2 == 0:  # kernel size should be odd
            ksize += 1
        kernel = np.ones((ksize, ksize), np.float32) / (ksize * ksize)  # uniform blur kernel, replace with desired PSF
        image = cv2.filter2D(image, -1, kernel)

    # apply gaussian blur
    if 'blur' in distortions:
        ksize = int(distortions['blur'])  # use the slider value for kernel size
        if ksize % 2 == 0:  # kernel size should be odd
            ksize += 1
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)

    # apply haze
    if 'haze' in distortions:
        intensity = distortions['haze']  # use the slider value for intensity
        haze = np.ones(image.shape, image.dtype) * 255  # this creates a white image
        image = cv2.addWeighted(image, 1 - intensity, haze, intensity, 0)  # blending the images

    # apply gradient
    if 'gradient' in distortions:
        intensity = distortions['gradient']
        gradient_mask = np.zeros_like(image, dtype=np.float32)
        for i in range(gradient_mask.shape[1]):
            gradient_mask[:, i, :] += i * intensity / gradient_mask.shape[1]
        gradient_mask = gradient_mask.astype(np.uint8)
        image = cv2.addWeighted(image, 1, gradient_mask, intensity, 0)

    # apply noise
    if 'noise' in distortions:
        mean = 0
        var = distortions['noise']
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, (image.shape[0], image.shape[1]))  # creating gaussian noise
        image = image + gaussian[:, :, np.newaxis]  # adding noise to image
        image = np.clip(image, 0, 255).astype(np.uint8)  # ensure values stay within [0, 255]
#Data Modeling
    if 'astigmatism' in distortions:
        intensity = random.randint(3,10)
        angle = random.randint(25,90)
        # Create an empty kernel
        kernel_size = int(intensity)
        kernel = np.zeros((kernel_size, kernel_size))

        # Convert angle to radians and calculate kernel coordinates
        angle = np.deg2rad(angle)  # convert degrees to radians
        x = int(kernel_size / 2 * np.cos(angle))  # x-coordinate
        y = int(kernel_size / 2 * np.sin(angle))  # y-coordinate

        # Modify the kernel at the specific angle
        cv2.line(kernel, (kernel_size // 2 - x, kernel_size // 2 - y),
                 (kernel_size // 2 + x, kernel_size // 2 + y), 1, thickness=1)

        # Normalize the kernel
        kernel = kernel / np.count_nonzero(kernel)

        # Apply the blur
        image = cv2.filter2D(image, -1, kernel)
#pinched optics Chromatic Abberation
    if 'feild_rotation' in distortions:
        rotation = random.randint(0,3)
        dilution = random.randint(1,5)
        offset = [0, 0]
        h, w = image.shape[:2]
        center = (w // 2 + offset[0], h // 2 + offset[1])

        # Compute the radius, circumference and steps
        radius = np.hypot(h, w) / 2
        circumference = 2 * np.pi * radius
        steps = int(circumference / dilution)

        accum_image = np.zeros_like(image, dtype=np.float32)

        for i in range(steps):
            angle = i * rotation / steps

            # Compute rotation matrix and apply warp affine
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            accum_image += rotated / steps
        image = accum_image

    if 'trailing' in distortions:
        length = random.randint(0,20)
        angle = random.randint(0, 90)
        dilution = 1
        h, w = image.shape[:2]

        # Compute the shift in x and y directions
        shift_x = length * np.cos(np.radians(angle))
        shift_y = length * np.sin(np.radians(angle))
        shift = (shift_x, shift_y)

        # Calculate the number of steps
        total_length = np.hypot(*shift)
        steps = int(total_length / dilution)

        accum_image = np.zeros_like(image, dtype=np.float32)

        for i in range(steps):
            dx = i * shift_x / steps
            dy = i * shift_y / steps

            # Compute transformation matrix and apply warp affine
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

            # Inpaint the shifted image
            mask = np.all(shifted == [0, 0, 0], axis=2).astype(np.uint8)
            inpainted = cv2.inpaint(shifted, mask, 3, cv2.INPAINT_TELEA)

            accum_image += inpainted / steps

    return image


def process_image(filename, distortion_ranges):
    # read image
    img_path = os.path.join(input_dir, filename)
    img = cv2.imread(img_path)

    # ensure image is not too small
    if img.shape[0] < min_height or img.shape[1] < min_width:
        print(f"Skipping file: {filename} - Image size is smaller than minimum size.")
        return

    for i in range(num_images):
        # crop image to desired size
        y = random.randint(0, img.shape[0] - min_height)
        x = random.randint(0, img.shape[1] - min_width)
        cropped_img = img[y:y + min_height, x:x + min_width]

        # Randomly flip the image (horizontally or vertically)
        flip_type = random.choice([-1, 0, 1]) # -1: flip vertically and horizontally, 0: flip vertically, 1: flip horizontally
        cropped_img = cv2.flip(cropped_img, flip_type)

        # Randomly rotate the image (in steps of 90 degrees)
        num_rotations = random.choice([0, 1, 2, 3]) # number of times to rotate 90 degrees clockwise
        cropped_img = np.rot90(cropped_img, num_rotations)

        # save validation image (non-distorted)
        validation_img_path = os.path.join(validation_dir, f"{os.path.splitext(filename)[0]}_{i + 1}.png")
        cv2.imwrite(validation_img_path, cropped_img)

        # generate distortions for this image
        distortions = {}
        for key, value_range in distortion_ranges.items():
            distortions[key] = random.uniform(*value_range)

        # distort and save training image
        distorted_img = apply_distortions(cropped_img.copy(), distortions)
        train_img_path = os.path.join(train_dir, f"{os.path.splitext(filename)[0]}_{i + 1}.png")
        cv2.imwrite(train_img_path, distorted_img)

        # save labels
        label_file_path = os.path.join(label_dir, f"{os.path.splitext(filename)[0]}_{i + 1}.txt")
        with open(label_file_path, 'w') as outfile:
            json.dump(distortions, outfile)


# process each image in the input directory
distortion_ranges = {'blur': (5, 15),'astigmatism':(11,13)}  # specify the range for each distortion
for filename in os.listdir(input_dir):
    try:
        print('processing: '+str(filename))
        process_image(filename, distortion_ranges)
    except Exception as e:
        print(e)
        continue
