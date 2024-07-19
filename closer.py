# Importing necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
import os
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from PIL import Image
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbMetricsLogger
from tensorflow.keras.layers import PReLU,Lambda, Layer, Subtract, Multiply, Add, LeakyReLU, PReLU, ELU, Dense, Reshape, GlobalAveragePooling2D, Input, Conv2DTranspose, Conv2D, BatchNormalization, Dropout, MaxPooling2D,Concatenate,UpSampling2D
from tensorflow.keras.applications.vgg19 import VGG19
from sklearn.model_selection import train_test_split
from tensorflow.keras.regularizers import l1_l2
from astropy.stats import sigma_clipped_stats
import cv2
from photutils.detection import find_peaks
from skimage import color, data, restoration
from astropy.table import Table
from astropy.nddata import NDData
from skimage.feature import peak_local_max
from skimage.morphology import disk
from photutils.psf import EPSFBuilder
from photutils.psf import extract_stars
from scipy.optimize import curve_fit
from tensorflow.keras.models import load_model
tf.config.run_functions_eagerly(True)
def gaussian_function(x, amplitude, mean, stddev):
    return amplitude * tf.math.exp(-(x - mean)**2 / (2 * stddev**2))
from keras.applications.vgg16 import VGG16
"""
vgg = VGG16(include_top=False, weights='imagenet')
for layer in vgg.layers:
    layer.trainable = False
vgg_features = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
def perceptual_loss(y_true, y_pred):
    return K.mean(K.square(vgg_features(y_true) - vgg_features(y_pred)))

"""
# Load pre-trained ResNet50 model + higher level layers
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

# Ensure the model is non-trainable
for layer in base_model.layers:
    layer.trainable = False

# Get specific layer output for perceptual loss

# Define SSIM loss function
def ssim_loss(y_true, y_pred):
    computed_loss = tf.cond(
        tf.size(y_pred) == 0,
        lambda: tf.constant(0.0, dtype=tf.float32),
        lambda: 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
    )
    return computed_loss

# Define combined loss
def combined_losskahsdfk(y_true, y_pred, alpha=0.7, beta=0.3):
    L_perceptual = perceptual_loss(y_true, y_pred)
    L_ssim = ssim_loss(y_true, y_pred)
    return alpha * L_perceptual + beta * L_ssim

base_vgg = VGG19(weights='imagenet', include_top=False)

# Specify the layers you want to extract features from
layer_names = ['block1_conv2']

# Create a model that will return these outputs, given the base_vgg model input
outputs = [base_vgg.get_layer(name).output for name in layer_names]
vgg_model = Model([base_vgg.input], outputs)

# Freeze the layers
for layer in vgg_model.layers:
    layer.trainable = False

def perceptual_loss(y_true, y_pred):
    y_true_features = vgg_model(y_true)
    y_pred_features = vgg_model(y_pred)

    losses = []
    for y_true_feature, y_pred_feature in zip(y_true_features, y_pred_features):
        losses.append(tf.reduce_mean(tf.square(y_true_feature - y_pred_feature)))

    return tf.reduce_sum(losses)

class SharpenFactorLayer(Layer):
    def __init__(self, initial_value=1.0, **kwargs):
        super(SharpenFactorLayer, self).__init__(**kwargs)
        self.sharpen_factor = self.add_weight(shape=(), initializer=tf.constant_initializer(initial_value),
                                              trainable=True, name='sharpen_factor')

    def call(self, inputs):
        sharpen_factor_expanded = tf.expand_dims(self.sharpen_factor, axis=-1)
        return inputs * sharpen_factor_expanded
class BoxBlurInversionLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(BoxBlurInversionLayer, self).__init__()
        self.kernel_size = tf.Variable(initial_value=tf.constant(3.0), trainable=True, dtype=tf.float32)

    def build(self, input_shape):
        self.kernel_size_int = tf.cast(tf.round(self.kernel_size), dtype=tf.int32)

    def call(self, inputs):
        kernel_weights = tf.ones([self.kernel_size_int, self.kernel_size_int, inputs.shape[-1], 1], dtype=inputs.dtype)
        inverted = tf.nn.depthwise_conv2d(inputs, kernel_weights, strides=[1, 1, 1, 1], padding='SAME')
        unblurred = inputs - inverted
        return unblurred
class TrainableMaskingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TrainableMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.mask = self.add_weight(name='mask', shape=(1, 1, num_channels), initializer='ones', trainable=True)

    def call(self, inputs):
        masked_inputs = inputs * self.mask
        return masked_inputs
# Setting hyperparameters and paths
#mirrored_strategy=tf.distribute.MirroredStrategy()
batch_size = 4
epochs = 300
learning_rate = 0.0001
data_directory_path = r"C:\Users\Administrator\Downloads\EverythingNeeded\Models"
data_dir_train = r"C:\Users\Administrator\Downloads\EverythingNeeded\Train"
data_dir_validation = r"C:\Users\Administrator\Downloads\EverythingNeeded\Validation"
min_width = 512
min_height = 512
color_mode = "rgb"
Image.MAX_IMAGE_PIXELS = None  # No limit to image size
def preprocess_image(img, target_size=(min_width, min_height), color_mode='rgb'):
    height, width, _ = img.shape

    # find the larger dimension
    max_dim = max(height, width)

    # find scale factor
    scale_factor = target_size[0] / max_dim

    # calculate new dimensions
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)

    # resize the image so the larger dimension fits the target size
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # pad the image to fit the target size
    delta_w = target_size[1] - new_width
    delta_h = target_size[0] - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    # convert the image to RGB or grayscale
    if color_mode == 'rgb':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode == 'gray':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def preprocess_image_2(img):
    # resize the image if it's smaller than the minimum size
    img = img.resize((min(2000, img.width), min(2000, img.height)))

    # convert the image to RGB or grayscale
    img = img.convert(color_mode.upper())

    return np.array(img)


def pad_images(images):
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    padded_images = []
    for image in images:
        pad_height = max_height - image.shape[0]
        pad_width = max_width - image.shape[1]
        padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)), mode='constant')
        padded_images.append(padded_image)
    return np.array(padded_images)


def equalize_and_normalize_hist(img):
    # Compute the maximum pixel value based on the data type of the image
    max_pixel_value = np.iinfo(img.dtype).max

    # Reshape the image to a 1D array.
    img_flat = img.flatten()

    # Get the histogram of the image.
    hist, bins = np.histogram(img_flat, max_pixel_value+1, [0,max_pixel_value+1])

    # Compute the cumulative distribution function of the histogram.
    cdf = hist.cumsum()

    # Mask all zeros in the cumulative distribution function.
    cdf_m = np.ma.masked_equal(cdf, 0)

    # Perform the histogram equalization.
    cdf_m = (cdf_m - cdf_m.min()) * max_pixel_value / (cdf_m.max() - cdf_m.min())

    # Fill masked pixels with zero and cast the result to the appropriate integer datatype.
    cdf = np.ma.filled(cdf_m, 0).astype(img.dtype)

    # Use the cumulative distribution function as a lookup table to equalize the pixels in the image.
    img_eq_flat = cdf[img_flat]

    # Reshape the equalized array back into the original 3D shape.
    img_eq = img_eq_flat.reshape(img.shape)

    # Normalize the equalized image to the range [0,1].
    img_norm = img_eq / max_pixel_value

    return img_norm


def open_image_as_np_array(image_path):
    # cv2.IMREAD_UNCHANGED ensures that the image's bit depth is preserved
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # cv2 reads images in BGR format, convert it to RGB
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def load_images(image_dir):
    print(f"Loading images from {image_dir}")
    images = []
    all_files = os.listdir(image_dir)
    num_files = len(all_files)
    for idx, filename in enumerate(all_files):
        try:
            im = Image.open(os.path.join(image_dir, filename))
            im = preprocess_image(im)
            im = im/255
            images.append(im)
            if idx % 50 == 0 or idx == num_files - 1: # Print a loading message every 50 images or at the end
                print(f"Loaded {idx + 1} out of {num_files} images")
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    print(f"Images from {image_dir} loaded.")
    return pad_images(images)
def preprocess_image(img):
    img = img.resize((min(min_width, img.width), min(min_height, img.height)))
    img = img.convert(color_mode.upper())
    return np.array(img)

def pad_images(images):
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    padded_images = []
    for image in images:
        pad_height = max_height - image.shape[0]
        pad_width = max_width - image.shape[1]
        padded_image = np.pad(image, ((0, pad_height), (0, pad_width), (0, 0)))
        padded_images.append(padded_image)
    return np.array(padded_images)
"""
@tf.function
def train_step(input_images, target_images):
    with tf.GradientTape() as tape:
        # Forward pass
        generated_images = model(input_images)

        # Compute the loss
        loss = perceptual_loss(target_images, generated_images)

    # Compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss
"""


"""
@tf.function
def train_step(input_images, target_images):
    with tf.GradientTape() as tape:
        # Forward pass
        generated_images = model(input_images)

        # Compute the loss
        loss = perceptual_loss(target_images, generated_images)

    # Compute gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss
"""
def calc_ssim_loss(y_true, y_pred):
    ssim = tf.image.ssim(y_true, y_pred, max_val=1.0)
    return 1.0 - ssim  # Return 1.0 minus SSIM to create a minimization loss

def perceptual_loss(y_true, y_pred):
    mse_loss_obj = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
    mse_loss = mse_loss_obj(y_true, y_pred)

    ssim_loss = calc_ssim_loss(y_true, y_pred)

    # Reduce mse_loss to have the same shape as ssim_loss (which is [batch_size])
    mse_loss_reduced = tf.reduce_mean(mse_loss, axis=[1, 2])

    total_loss = mse_loss_reduced + ssim_loss

    computed_loss = tf.cond(
        tf.size(y_pred) == 0,
        lambda: tf.constant(0.0, dtype=tf.float32),
        lambda: tf.reduce_mean(total_loss)
    )

    return computed_loss

def squeeze_excite_block(input, ratio=16):
    channels = input.shape[-1]
    se = GlobalAveragePooling2D()(input)
    se = Reshape((1, 1, channels))(se)

    # Add a Convolutional layer to capture spatial information
    se = Conv2D(channels // ratio, kernel_size=1, activation='relu')(se)

    # Modify the attention mechanism to focus on nebulae
    se = Conv2D(channels, kernel_size=1, activation='sigmoid')(se)

    return Multiply()([input, se])
def star_attention_block(input, ratio=16):
    channels = input.shape[-1]
    se = GlobalAveragePooling2D()(input)
    se = Reshape((1, 1, channels))(se)

    # Add a Convolutional layer to capture spatial information
    se = Conv2D(channels // ratio, kernel_size=1, activation='relu')(se)

    # Modify the attention mechanism to focus on stars
    se = Conv2D(channels, kernel_size=1, activation='sigmoid')(se)

    return Multiply()([input, se])
def calculate_psf(star_patches):
    # Calculate PSF by taking the average of star patches
    psf = tf.reduce_mean(star_patches, axis=0)
    return psf
@tf.function
def extract_stars(image, star_coords, patch_size):
    y_coords = star_coords[:, 1]
    x_coords = star_coords[:, 2]

    patches = tf.image.extract_patches(
        image, sizes=[1, patch_size, patch_size, 1],
        strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME'
    )

    y_coords = tf.expand_dims(y_coords, axis=1)
    x_coords = tf.expand_dims(x_coords, axis=1)

    y_coords = tf.repeat(y_coords, patch_size * patch_size, axis=1)
    x_coords = tf.repeat(x_coords, patch_size * patch_size, axis=1)

    y_coords = tf.reshape(y_coords, (-1,))
    x_coords = tf.reshape(x_coords, (-1,))

    indices = tf.stack([y_coords, x_coords], axis=1)

    star_patches = tf.gather_nd(patches, indices)
    star_patches = tf.reshape(star_patches, (-1, patch_size, patch_size, 3, 3))

    return star_patches
def custom_peak_local_max(image, threshold_rel, max_peaks=50):
    thresh=tf.reduce_mean(image)
    peak_coords = tf.where(image > thresh)
    return peak_coords[:max_peaks, :]

def visualize_star_patches(star_patches):
    num_patches = star_patches.shape[0]
    fig, axes = plt.subplots(1, num_patches, figsize=(15, 5))

    for i in range(num_patches):
        patch = star_patches[i]
        axes[i].imshow(patch)
        axes[i].axis('off')

    plt.show()



def adaptive_pooling(x, target_size):
    return tf.image.resize(x, target_size, method=tf.image.ResizeMethod.AVERAGE)
def spatial_attention(input_tensor, alpha=0.1):
    # Create a convolutional layer to compute attention scores
    attention = Conv2D(1, (1, 1), activation='sigmoid', use_bias=False)(input_tensor)

    # Multiply the input tensor by the attention scores
    x = Multiply()([input_tensor, attention])

    # Apply leaky ReLU for non-linearity
    x = LeakyReLU(alpha=alpha)(x)

    return x
def psf_extract(input_tensor):
    alpha = 0.2
    x = Conv2D(64, 3, strides=1, padding='same')(input_tensor)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(64, 3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)

    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(128, 3, strides=2, padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)

    x = Conv2D(256,3,strides=1,padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(256,3,strides=2,padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(512,3,strides=1,padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(512,3,strides=2,padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(512,3,strides=1,padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(256,3,strides=2,padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(256,3,strides=1,padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)

    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(128, 3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)

    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(64, 3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=alpha)(x)

    x = Conv2D(1, 3, strides=1, activation='linear', padding='same',name='psf_output')(x)
    return x

class RichardsonLucyDeconvLayer(tf.keras.layers.Layer):
    def __init__(self, num_iter=3, **kwargs):
        super(RichardsonLucyDeconvLayer, self).__init__(**kwargs)
        self.num_iter = num_iter

    def call(self, inputs):
        image, psf = inputs
        avg_psf = tf.reduce_mean(psf, axis=0)
        avg_psf = tf.squeeze(avg_psf, axis=-1)  # Now shape is [height, width]

        # Extend to [height, width, 1, 1]
        avg_psf = tf.expand_dims(tf.expand_dims(avg_psf, axis=-1), axis=-1)

        # Tile to [height, width, 3, 3]
        psf = tf.tile(avg_psf, [1, 1, 3, 3])

        decon_image = image
        for _ in range(self.num_iter):
            conv = tf.nn.conv2d(image, psf, strides=[1, 1, 1, 1], padding='SAME')
            relative_blur = image / (conv + 1e-10)  # added small value to prevent division by zero
            decon_image *= tf.nn.conv2d(relative_blur, psf, strides=[1, 1, 1, 1], padding='SAME')

        return decon_image


def create_model(dropout_rate=0.2, l1_reg=0.01, l2_reg=0.01):
    deconv_layer = RichardsonLucyDeconvLayer()
    variable = 0.1
    inputs = Input((None,None,3))
    psf = psf_extract(inputs)


    decon_image = deconv_layer([inputs, psf])

    #Encoder based, but not an Encoder
    conv1 = Conv2D(64,3,strides=1,padding='same')(decon_image)
    conv1 = LeakyReLU(alpha=0.2)(conv1)
    conv1 = Dropout(variable)(conv1)

    conv2 = Conv2D(128,3,strides=1,padding='same')(conv1)
    conv2 = LeakyReLU(alpha=0.2)(conv2)
    conv2 = Dropout(variable)(conv2)

    conv3 = Conv2D(256,3,strides=1,padding='same')(conv2)
    conv3 = LeakyReLU(alpha=0.2)(conv3)
    conv3 = Dropout(variable)(conv3)

    #Latent Space "Latent"
    conv4 = Conv2D(512,3,strides=1,padding='same')(conv3)
    conv4 = LeakyReLU(alpha=0.2)(conv4)

    conv5 = Conv2D(1028,3,strides=1,padding='same')(conv4)
    conv5 = LeakyReLU(alpha=0.2)(conv5)

    conv6 = Conv2D(2048,3,strides=1,padding='same')(conv5)
    conv6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = Conv2D(2048,3,strides=1,padding='same')(conv5)
    conv6 = LeakyReLU(alpha=0.2)(conv6)

    conv6 = Conv2D(1028,3,strides=1,padding='same')(conv5)
    conv6 = LeakyReLU(alpha=0.2)(conv6)
    conv6 = Add()([conv6,conv5])


    conv7 = Conv2D(512,3,strides=1,padding='same')(conv6)
    conv7 = LeakyReLU(alpha=0.2)(conv7)
    conv7 = Add()([conv7,conv4])

    #Decoder but not really
    conv8 = Conv2D(256,3,strides=1,padding='same')(conv7)
    conv8 = LeakyReLU(alpha=0.2)(conv8)
    conv8 = Add()([conv3,conv8])
    conv8 = Dropout(variable)(conv8)

    conv9 = Conv2D(128,3,strides=1,padding='same')(conv8)
    conv9 = LeakyReLU(alpha=0.2)(conv9)
    #conv9 = Add()([conv9,conv2])
    conv9 = Dropout(variable)(conv9)

    conv10 = Conv2D(64,3,strides=1,padding='same')(conv9)
    conv10 = LeakyReLU(alpha=0.2)(conv10)
    conv10 = Dropout(variable)(conv10)
    #conv10 = Add()([conv10,conv1])

    outputs = Conv2D(3, 3,strides=1,padding='same', activation='sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
train_dataset = load_images(data_dir_train)
val_dataset = load_images(data_dir_validation)
# Create the model inside the strategy's scope
# Define and compile your model
# Define and compile your model

def lossfunc(y_true,y_pred):
    loss = tf.abs(tf.reduce_mean(y_true - y_pred))
    return loss
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
# First, define the strategy
strategy = tf.distribute.MirroredStrategy(
     cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=combined_losskahsdfk)

# Split your data into train and validation sets
os.environ["NCCL_DEBUG"] = "INFO"

# Create train and validation datasets
with tf.device('CPU'):
    train_images, val_images, train_labels, val_labels = train_test_split(train_dataset, val_dataset, test_size=0.2)
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


# Checkpoint setup remains the same
checkpoint_dir = r"C:\Users\Administrator\Downloads\EverythingNeeded"
current_datetime = datetime.now()
datetime_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_file = os.path.join(checkpoint_dir, f"model_{datetime_string}")
checkpoint_callback = ModelCheckpoint(checkpoint_file, save_best_only=True, monitor='val_loss', mode='min')

early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=1e-4,
    patience=3,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False,
)

lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
#wandb.init(config={"bs":12})
#WandbMetricsLogger()
# Train the model
learning_rate_scheduler = LearningRateScheduler(
    schedule = lambda epoch:learning_rate * (0.5**(epoch//5))
)
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=[lr_scheduler, checkpoint_callback, early_stopping_callback])

# Reset the iterators at the end of each epoch
#train_dataset_iterator = iter(train_dataset)
#val_dataset_iterator = iter(val_dataset)

# Training loop with iterators

# Make predictions

# Format current date and time and create a filename
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_filename = 'vikX' + current_time

# Save the model
model.save(os.path.join(data_directory_path, model_filename))

# Plotting the training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("learning_curve.png")
plt.show()
