import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import numpy as np
from tensorflow.keras.layers import concatenate,Activation,BatchNormalization,Lambda, Layer, Subtract, Multiply, Add, LeakyReLU, PReLU, ELU, Dense, Reshape, GlobalAveragePooling2D, Input, Conv2DTranspose, Conv2D, BatchNormalization, Dropout, MaxPooling2D,Concatenate,UpSampling2D
#from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from keras.applications.vgg16 import VGG16
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

# Ensure the model is non-trainable
for layer in base_model.layers:
    layer.trainable = False
resnet_features = Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block6_out').output)

def perceptual_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(resnet_features(y_true) - resnet_features(y_pred)))
min_width = 2000  # you can change this to your preferred minimum width
min_height = 1300
color_mode = "rgb"
class SharpenFactorLayer(Layer):
    def __init__(self, initial_value=1.0, **kwargs):
        super(SharpenFactorLayer, self).__init__(**kwargs)
        self.sharpen_factor = self.add_weight(shape=(), initializer=tf.constant_initializer(initial_value),
                                              trainable=True, name='sharpen_factor')

    def call(self, inputs):
        sharpen_factor_expanded = tf.expand_dims(self.sharpen_factor, axis=-1)
        return inputs * sharpen_factor_expanded
class BoxBlurInversionLayer(tf.keras.layers.Layer):
    def __init__(self, name=None, trainable=True, dtype=tf.float32):
        super(BoxBlurInversionLayer, self).__init__()
        self.kernel_size = tf.Variable(initial_value=3.0, dtype=tf.float32, trainable=True)

    def call(self, inputs):
        kernel_size_int = tf.cast(tf.round(self.kernel_size), dtype=tf.int32)
        kernel_weights = tf.ones([kernel_size_int, kernel_size_int, inputs.shape[-1], 1], dtype=inputs.dtype)
        inverted = tf.nn.depthwise_conv2d(inputs, kernel_weights, strides=[1, 1, 1, 1], padding='SAME')
        unblurred = inputs - inverted
        return unblurred
# Register the custom layer
custom_objects = {'SharpenFactorLayer': SharpenFactorLayer, 'BoxBlurInversionLayer':BoxBlurInversionLayer}

# Load the model
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
def preprocess_image(img):
    # resize the image if it's smaller than the minimum size
    img = img.resize((512, 512))

    # convert the image to RGB or grayscale
    img = img.convert(color_mode.upper())
    return np.array(img)
def preprocess_image_2(img):
    # resize the image if it's smaller than the minimum size
    img = img.resize((512, 512))

    # convert the image to RGB or grayscale
    img = img.convert(color_mode.upper())
    return img
class TrainableMaskingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TrainableMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.mask = self.add_weight(name='mask', shape=(1, 1, num_channels), initializer='ones', trainable=True)

    def call(self, inputs):
        masked_inputs = inputs * self.mask
        return masked_inputs
def psf_extract(input_tensor):
    x = Conv2D(64, 3, strides=1, padding='same',activation='relu')(input_tensor)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, 3, strides=1, padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(256, 3, strides=1, padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(128, 3, strides=1, padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(64, 3, strides=1, padding='same',activation='relu')(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(3, 3, strides=1, activation='linear', padding='same')(x)
    return x

class RichardsonLucyDeconvLayer(tf.keras.layers.Layer):
    def __init__(self, num_iter=10, **kwargs):
        super(RichardsonLucyDeconvLayer, self).__init__(**kwargs)
        self.num_iter = num_iter

    def call(self, inputs):
        image, psf = inputs
        average_psf = tf.reduce_mean(psf, axis=0)  # This will be shape [8, 8, 3]
        # Now, expand the last dimension to get [8, 8, 3, 3]
        final_psf = tf.expand_dims(average_psf, axis=-1)  # This is shape [8, 8, 3, 1]
        final_psf = tf.tile(final_psf, [1, 1, 1, 3])
        psf = final_psf
        decon_image = image
        for _ in range(self.num_iter):
            conv = tf.nn.conv2d(image, psf, strides=[1, 1, 1, 1], padding='SAME')
            relative_blur = image / conv
            decon_image *= tf.nn.conv2d(relative_blur, psf, strides=[1, 1, 1, 1], padding='SAME')
        return decon_image

# Load pre-trained ResNet50 model + higher level layers
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

# Ensure the model is non-trainable
for layer in base_model.layers:
    layer.trainable = False

# Get specific layer output for perceptual loss
resnet_features = tf.keras.Model(inputs=base_model.input, outputs=base_model.get_layer('conv4_block6_out').output)

# Define the perceptual loss function
def perceptual_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(resnet_features(y_true) - resnet_features(y_pred)))

# Define SSIM loss function
def ssim_loss(y_true, y_pred):
    return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

# Define combined loss
def combined_losskahsdfk(y_true, y_pred, alpha=0.8, beta=0.2):
    L_perceptual = perceptual_loss(y_true, y_pred)
    L_ssim = ssim_loss(y_true, y_pred)
    return alpha * L_perceptual + beta * L_ssim

class BinaryMaskingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BinaryMaskingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.threshold = 0.3
        super(BinaryMaskingLayer, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        mask = tf.where(inputs >= self.threshold, 0.0, 1.0)
        inverted_mask = 1.0 - mask
        mask_mean = tf.reduce_mean(mask, axis=-1, keepdims=True)
        # Repeat the mean channel 3 times to have the same value for all channels
        mask = tf.tile(mask_mean, [1, 1, 1, 1])
        invmask_mean = tf.reduce_mean(inverted_mask, axis=-1, keepdims=True)
        # Repeat the mean channel 3 times to have the same value for all channels
        inverted_mask = tf.tile(invmask_mean, [1, 1, 1, 3])
        masked_inputs = inputs * mask
        return masked_inputs, mask, inverted_mask
def create_model(dropout_rate=0.2, l1_reg=0.01, l2_reg=0.01):
    maskLayer=BinaryMaskingLayer()
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

    #Latent Space "Latent"
    conv4 = Conv2D(512,3,strides=1,padding='same')(conv3)
    conv4 = LeakyReLU(alpha=0.2)(conv4)

    conv5 = Conv2D(1024,3,strides=1,padding='same')(conv4)
    conv5 = LeakyReLU(alpha=0.2)(conv5)

    conv7 = Conv2D(512,3,strides=1,padding='same')(conv5)
    conv7 = LeakyReLU(alpha=0.2)(conv7)

    #Decoder but not really
    conv8 = Conv2D(256,3,strides=1,padding='same')(conv7)
    conv8 = LeakyReLU(alpha=0.2)(conv8)
    conv8 = Add()([conv3,conv8])
    conv8 = Dropout(variable)(conv8)

    conv9 = Conv2D(128,3,strides=1,padding='same')(conv8)
    conv9 = LeakyReLU(alpha=0.2)(conv9)
    conv9 = Add()([conv9,conv2])
    conv9 = Dropout(variable)(conv9)

    conv10 = Conv2D(64,3,strides=1,padding='same')(conv9)
    conv10 = LeakyReLU(alpha=0.2)(conv10)
    conv10=Add()([conv10,conv1])

    outputs = Conv2D(3, 3,strides=1,padding='same', activation='sigmoid')(conv10)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Load the mode
import matplotlib.pyplot as plt
sample_image_path=r"C:\Users\Administrator\Downloads\EverythingNeeded\Train\opo0315d_3.png"
model=create_model()
model_path=r"C:\Users\Administrator\Downloads\EverythingNeeded\model_2024-02-07_20-31-11"
custom_objects = {'combined_losskahsdfk': combined_losskahsdfk}

model = load_model(model_path, custom_objects=custom_objects)
model.summary()
deconv_layer = model.get_layer('richardson_lucy_deconv_layer')

# Step 3: Modify the desired attribute
deconv_layer.num_iter = 7 # replace 'new_num_iter_value' with your desired number
sample_image = Image.open(sample_image_path)
show_image=preprocess_image_2(sample_image)
show_image.save('original2.png')
show_image.show()
sample_image = preprocess_image(sample_image)
sample_image = sample_image / 255.0
sample_image = np.expand_dims(sample_image, axis=0)
output_image = model.predict(sample_image)
output_image = output_image.squeeze() * 255.0
output_image = Image.fromarray(output_image.astype(np.uint8))
output_image.save('after2.png')
output_image.show()




# Add spacing between subplots
