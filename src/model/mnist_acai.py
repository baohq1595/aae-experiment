import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import keras.backend as K
from collections import defaultdict

import matplotlib.pyplot as plt

import numpy as np
import tqdm
import os
from PIL import Image
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import seaborn as sns
# import pandas as pd

import sys
sys.path.append('.')

from src.utils.visualize import visualize_latent_space


class ACAI():
    def __init__(self, img_shape=(28,28), latent_dim=32, disc_reg_coef=0.2, ae_reg_coef=0.5, dropout=0.0):
        self.latent_dim = latent_dim
        self.ae_optim = Adam(0.0001)
        self.d_optim = Adam(0.0001)
        self.img_shape = img_shape
        self.dropout = dropout
        self.disc_reg_coef = disc_reg_coef
        self.ae_reg_coef = ae_reg_coef
        self.intitializer = tf.keras.initializers.VarianceScaling(
                            scale=0.2, mode='fan_in', distribution='truncated_normal')
        self.initialize_models(self.img_shape, self.latent_dim)

    def initialize_models(self, img_shape, latent_dim):
        self.encoder = self.build_encoder(img_shape, latent_dim)
        self.decoder = self.build_decoder(latent_dim, img_shape)
        self.discriminator = self.build_discriminator(latent_dim, img_shape)
        
        img = Input(shape=img_shape)
        latent = self.encoder(img)
        res_img = self.decoder(latent)
        
        self.autoencoder = Model(img, res_img)
        discri_out = self.discriminator(img)


    def build_encoder(self, img_shape, latent_dim):
        encoder = Sequential(name='encoder')
        encoder.add(Flatten(input_shape=img_shape))
        encoder.add(Dense(1000, activation=tf.nn.leaky_relu, kernel_initializer=self.intitializer))
        encoder.add(Dropout(self.dropout))
        encoder.add(Dense(1000, activation=tf.nn.leaky_relu, kernel_initializer=self.intitializer))
        encoder.add(Dropout(self.dropout))
        # encoder.add(Dense(2000, activation='relu'))
        # encoder.add(Dropout(self.dropout))
        # encoder.add(Dense(2000, activation='relu'))
        # encoder.add(Dropout(self.dropout))
        encoder.add(Dense(latent_dim))
        
        encoder.summary()
        return encoder
    
    def build_decoder(self, latent_dim, img_shape):
        decoder = Sequential(name='decoder')
        decoder.add(Dense(1000, input_dim=latent_dim, activation=tf.nn.leaky_relu, kernel_initializer=self.intitializer))
        decoder.add(Dropout(self.dropout))
        decoder.add(Dense(1000, activation=tf.nn.leaky_relu, kernel_initializer=self.intitializer))
        decoder.add(Dropout(self.dropout))
        # decoder.add(Dense(2000, activation='relu'))
        # decoder.add(Dropout(self.dropout))
        # decoder.add(Dense(2000, activation='relu'))
        # decoder.add(Dropout(self.dropout))
        decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
        decoder.add(Reshape(img_shape))
        
        decoder.summary()
        return decoder

    def build_discriminator(self, latent_dim, img_shape):
        discriminator = Sequential(name='discriminator')
        discriminator.add(Flatten(input_shape=img_shape))
        discriminator.add(Dense(1000, activation=tf.nn.leaky_relu, kernel_initializer=self.intitializer))
        discriminator.add(Dropout(self.dropout))
        discriminator.add(Dense(1000, activation=tf.nn.leaky_relu, kernel_initializer=self.intitializer))
        discriminator.add(Dropout(self.dropout))
        # discriminator.add(Dense(2000, activation='relu'))
        # discriminator.add(Dropout(self.dropout))
        # discriminator.add(Dense(2000, activation='relu'))
        # discriminator.add(Dropout(self.dropout))
        discriminator.add(Dense(latent_dim))

        # discriminator.add(Reshape((-1,)))
        discriminator.add(Lambda(lambda x: tf.reduce_mean(x, axis=1)))
        
        discriminator.summary()
        return discriminator

    def save(self, save_path):
        ae_path = os.path.join(save_path, ' ')
        d_path = os.path.join(save_path, 'discriminator')

        self.encoder.save(os.path.join(ae_path, 'encoder'))
        self.decoder.save(os.path.join(ae_path, 'decoder'))
        self.discriminator.save(d_path)

    def load(self, load_path):
        ae_path = os.path.join(load_path, 'autoencoder')
        d_path = os.path.join(load_path, 'discriminator')

        # Load encoder/decoder
        self.encoder = tf.keras.models.load_model(os.path.join(ae_path, 'encoder'))
        self.decoder = tf.keras.models.load_model(os.path.join(ae_path, 'decoder'))

        # Reset module autoencoder
        img = Input(shape=self.img_shape)
        latent = self.encoder(img)
        res_img = self.decoder(latent)
        self.autoencoder = Model(img, res_img)

        # Load discriminator
        self.discriminator = tf.keras.models.load_model(d_path)

def make_image_grid(imgs, shape=None, prefix=None, save_path=None, is_show=False):
    fig = plt.figure(figsize=[20,20])
    for index, img in enumerate(imgs):
        img = img.reshape(shape) if shape != img.shape else img
        ax = fig.add_subplot(10, 10, index + 1)
        ax.set_axis_off()
        ax.imshow(img, cmap='gray')
    if prefix != None and save_path != None:
        fig.savefig(os.path.join(save_path, prefix + '.png'))
        img = Image.open(os.path.join(save_path, prefix + '.png'))
    else:
        img = imgs
    if is_show:
        plt.show()
    plt.close(fig)

    return img

def flip_tensor(t):
    a, b = tf.split(x, 2, axis=0)
    return tf.concat([b, a], axis=0)

def avg_losses(total_losses):
    flatten_losses = defaultdict(list)
    for loss in total_losses:
        for kv in loss.items():
            flatten_losses[kv[0]].append(kv[1])

    avg_loss = {kv[0]: sum(kv[1]) / len(kv[1]) for kv in flatten_losses.items()}

    return avg_loss


@tf.function
def train_on_batch(x, y, model: ACAI):
    # Randomzie interpolated coefficient alpha
    # alpha = np.linspace(0, 0.5, x.shape[0]).reshape(-1,1,1,1)
    alpha = tf.random.uniform((x.shape[0], 1), 0, 1)
    alpha = 0.5 - tf.abs(alpha - 0.5)  # Make interval [0, 0.5]

    with tf.GradientTape() as ae_tape, tf.GradientTape() as d_tape:
        # Constructs non-interpolated latent space and decoded input
        latent = model.encoder(x, training=True)
        res_x = model.decoder(latent, training=True)

        ae_loss = tf.reduce_mean(tf.losses.mean_squared_error(tf.reshape(x, (x.shape[0], -1)), tf.reshape(res_x, (res_x.shape[0], -1))))

        inp_latent = alpha * latent + (1 - alpha) * latent[::-1]
        res_x_hat = model.decoder(inp_latent, training=False)

        pred_alpha = model.discriminator(res_x_hat, training=True)
        # pred_alpha = K.mean(pred_alpha, [1,2,3])
        temp = model.discriminator(res_x + model.disc_reg_coef * (x - res_x), training=True)
        # temp = K.mean(temp, [1,2,3])
        disc_loss_term_1 = tf.reduce_mean(tf.square(pred_alpha - alpha))
        disc_loss_term_2 = tf.reduce_mean(tf.square(temp))

        reg_ae_loss = model.ae_reg_coef * tf.reduce_mean(tf.square(pred_alpha))

        total_ae_loss = ae_loss + reg_ae_loss
        total_d_loss = disc_loss_term_1 + disc_loss_term_2

    # ae_var = []
    # ae_var.extend(model.encoder.trainable_variables)
    # ae_var.extend(model.decoder.trainable_variables)
    grad_ae = ae_tape.gradient(total_ae_loss, model.autoencoder.trainable_variables)
    grad_d = d_tape.gradient(total_d_loss, model.discriminator.trainable_variables)

    model.ae_optim.apply_gradients(zip(grad_ae, model.autoencoder.trainable_variables))
    model.d_optim.apply_gradients(zip(grad_d, model.discriminator.trainable_variables))

    return {
        'res_ae_loss': ae_loss,
        'reg_ae_loss': reg_ae_loss,
        'disc_loss': disc_loss_term_1,
        'reg_disc_loss': disc_loss_term_2

    }

def train(model: ACAI, x_train, y_train, x_test,
          batch_size, epochs=1000, save_interval=200,
          save_path='./images'):
    n_epochs = tqdm.tqdm_notebook(range(epochs))
    total_batches = x_train.shape[0] // batch_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in n_epochs:
        offset = 0
        losses = []
        random_idx = np.random.randint(0, x_train.shape[0], x_train.shape[0])
        for batch_iter in range(1):
            # Randomly choose each half batch
            imgs = x_train[offset:offset + batch_size,::] if (batch_iter < (total_batches - 1)) else x_train[offset:,::]
            offset += batch_size

            loss = train_on_batch(imgs, None, model)
            losses.append(loss)

        avg_loss = avg_losses(losses)
        # wandb.log({'losses': avg_loss})
            
        if epoch % save_interval == 0 or (epoch == epochs - 1):
            sampled_imgs = model.autoencoder(x_test[:100])
            res_img = make_image_grid(sampled_imgs.numpy(), (28,28), str(epoch), save_path)
            
            latent = model.encoder(x_train, training=False).numpy()
            latent_space_img = visualize_latent_space(latent, y_train, 10, is_save=True, save_path=f'./latent_space/{epoch}')
            # wandb.log({'res_test_img': [wandb.Image(res_img, caption="Reconstructed images")],
            #             'latent_space': [wandb.Image(latent_space_img, caption="Latent space")]})


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    acai = ACAI(dropout=0.5)
    train(model=acai,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            batch_size=4,
            epochs=5000,
            save_interval=200,
            save_path='./images')

    # Save weights
    acai.save('./weights')

    # Test interpolation
    # alpha values
    alpha_vals = np.linspace(1.0, 0.0, 11)

    # Get some samples
    left_column = x_test[:10,::]
    right_column = x_test[30:40,::]

    # Infer latent space
    left_latent = acai.encoder.predict(left_column)
    right_latent = acai.encoder.predict(right_column)

    # Mixing latents and decode samples
    samples = []
    for alpha in alpha_vals:
        alpha_array = np.full(left_latent.shape, alpha, dtype=np.float)
        alpha_array = tf.constant(alpha)
        mix_latent = alpha_array * left_latent + (1 - alpha_array) * right_latent
        mix_imgs = acai.decoder(mix_latent)
        samples.append(mix_imgs)

    _ = make_image_grid(samples, shape=(28, 28), prefix='acai_interpolate', save_path='.',  is_show=False)
    