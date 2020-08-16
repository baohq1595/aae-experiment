import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Reshape
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import keras.backend as K

import matplotlib.pyplot as plt

import os
import io
from PIL import Image
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# # from cuml.manifold import TSNE # If available, use it or
# # from tsnecuda import TSNE # If available, use it
# import seaborn as sns
# import pandas as pd

import numpy as np
import tqdm
import os

import sys
sys.path.append('.')

from src.utils.visualize import visualize_latent_space

class AdversarialAutoencoder():
    def __init__(self, img_shape=(28,28), latent_dim=8, dropout=0.0):
        self.latent_dim = latent_dim
        self.ae_optim = Adam(0.001)
        self.d_optim = Adam(0.001)
        self.img_shape = img_shape
        self.dropout = dropout
        self.initialize_models(self.img_shape, self.latent_dim)
        
    def initialize_models(self, img_shape, latent_dim):
        self.encoder = self.build_encoder(img_shape, latent_dim)
        self.decoder = self.build_decoder(latent_dim, img_shape)
        self.discriminator = self.build_discriminator(latent_dim)
        
        img = Input(shape=img_shape)
        latent = self.encoder(img)
        res_img = self.decoder(latent)
        
        self.autoencoder = Model(img, res_img)
        discri_out = self.discriminator(latent)
        
        self.generator = Model(img, discri_out)
        self.discriminator.compile(optimizer=self.d_optim,
                                    loss='binary_crossentropy')
        self.autoencoder.compile(optimizer=self.ae_optim,
                                    loss='mse')
        # generator_discriminator ~ only train generator
        for layer in self.discriminator.layers:
            layer.trainable = False
        self.generator.compile(optimizer=self.d_optim,
                                    loss='binary_crossentropy')
        
    
    def build_encoder(self, img_shape, latent_dim):
        encoder = Sequential()
        encoder.add(Flatten(input_shape=img_shape))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dropout(self.dropout))
        encoder.add(Dense(1000, activation='relu'))
        encoder.add(Dropout(self.dropout))
        encoder.add(Dense(latent_dim))
        
        encoder.summary()
        return encoder
    
    def build_decoder(self, latent_dim, img_shape):
        decoder = Sequential()
        decoder.add(Dense(1000, input_dim=latent_dim, activation='relu'))
        decoder.add(Dropout(self.dropout))
        decoder.add(Dense(1000, activation='relu'))
        decoder.add(Dropout(self.dropout))
        decoder.add(Dense(np.prod(img_shape), activation='sigmoid'))
        decoder.add(Reshape(img_shape))
        
        decoder.summary()
        return decoder
    
    def build_discriminator(self, latent_dim):
        discriminator = Sequential()
        discriminator.add(Dense(1000, input_dim=latent_dim, activation='relu'))
        discriminator.add(Dropout(self.dropout))
        discriminator.add(Dense(1000, activation='relu'))
        discriminator.add(Dropout(self.dropout))
        discriminator.add(Dense(1, activation='sigmoid'))
        
        discriminator.summary()
        return discriminator
    
    def samples(self, num_images=100):
        latents = 5 * np.random.normal(size=(num_images, self.latent_dim))
        imgs = self.decoder.predict(latents)
        
        return imgs

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        ae_path = os.path.join(save_path, 'autoencoder')
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
    

import os
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


from collections import defaultdict
def avg_losses(total_losses):
    flatten_losses = defaultdict(list)
    for loss in total_losses:
        for kv in loss.items():
            flatten_losses[kv[0]].append(kv[1])

    avg_loss = {kv[0]: sum(kv[1]) / len(kv[1]) for kv in flatten_losses.items()}

    return avg_loss

import tqdm
def train(model: AdversarialAutoencoder, x_train, y_train,
          batch_size, epochs=1000, save_interval=200,
          save_path='./images'):
    # latents = 5 * np.random.normal(size=(100, model.latent_dim))
    latents = 5 * np.random.uniform(-1, 1, size=(100, model.latent_dim))
    n_epochs = tqdm.tqdm_notebook(range(epochs))
    half_batch = int(batch_size / 2)
    total_batches = x_train.shape[0] // batch_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for epoch in n_epochs:
        losses = []
        offset = 0
        for batch_iter in range(1):
            # Randomly choose each half batch
            imgs = x_train[offset:offset + batch_size,::] if (batch_iter < (total_batches - 1)) else x_train[offset:,::]
            offset += batch_size
            # Randomly choose each half batch
            idx = np.random.randint(0, x_train.shape[0], half_batch)
            imgs = x_train[idx]
            
            # Train discriminator
            ## Get fake and real latent feature
            latent_fake = model.encoder.predict(imgs)
            latent_real = 5 * np.random.normal(size=(half_batch, model.latent_dim))
            d_real = np.ones((half_batch, 1))
            d_fake = np.zeros((half_batch, 1))
            
            ## train
            d_loss_real = model.discriminator.train_on_batch(latent_real, d_real)
            d_loss_fake = model.discriminator.train_on_batch(latent_fake, d_fake)
            d_loss = d_loss_real + d_loss_fake
            
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            
            # Train autoencoder
            ae_loss = model.autoencoder.train_on_batch(imgs, imgs)
            
            # Train generator
            g_real = np.ones((batch_size, 1))
            g_loss = model.generator.train_on_batch(imgs, g_real)
        
            loss = {
                'ae_loss': ae_loss,
                'g_loss': g_loss,
                'd_loss': d_loss
            }
            losses.append(loss)

        avg_loss = avg_losses(losses)
        # wandb.log({'losses': avg_loss})
        
        if epoch % save_interval == 0 or (epoch == epochs - 1):
            sampled_imgs = model.decoder(latents, training=False)
            res_img = make_image_grid(sampled_imgs.numpy(), (28,28), str(epoch), save_path)
            
            latent = model.encoder.predict(x_train)
            latent_space_img = visualize_latent_space(latent, y_train, 10, is_save=True, save_path=f'./latent_space/{epoch}.png')
            # wandb.log({'samples_imgs': [wandb.Image(res_img, caption="Sampled images")],
            #             'latent_space': [wandb.Image(latent_space_img, caption="Latent space")]})

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.
    x_test = x_test.astype(np.float32) / 255.
    aae = AdversarialAutoencoder(dropout=0.5)
    train(model=aae,
            x_train=x_train,
            y_train=y_train,
            batch_size=4,
            epochs=1,
            save_interval=50,
            save_path='./images')

    # Save weights
    aae.save('./weights')

    # Test interpolation
    # alpha values
    alpha_vals = np.linspace(1.0, 0.0, 11)

    # Get some samples
    left_column = x_test[:10,::]
    right_column = x_test[30:40,::]

    # Infer latent space
    left_latent = aae.encoder.predict(left_column)
    right_latent = aae.encoder.predict(right_column)

    # Mixing latents and decode samples
    samples = []
    for alpha in alpha_vals:
        alpha_array = np.full(left_latent.shape, alpha, dtype=np.float)
        alpha_array = tf.constant(alpha)
        mix_latent = alpha_array * left_latent + (1 - alpha_array) * right_latent
        mix_imgs = aae.decoder(mix_latent)
        samples.append(mix_imgs)

    _ = make_image_grid(samples, shape=(28, 28), prefix='aae_interpolate', save_path='.',  is_show=False)





