# %% --------------------------------------- Load Packages -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from numpy.random import shuffle
import cv2
# %% --------------------------------------- Load Models ---------------------------------------------------------------
# Carga tu dataset
train_data = np.load('dataset.npz')
images = train_data['x']
labels = train_data['y']
#latent=0
# Set channel
channel = images.shape[-1]

# to 64 x 64 x channel
real = np.ndarray(shape=(images.shape[0], 64, 64, channel))
for i in range(images.shape[0]):
    real[i] = cv2.resize(images[i], (64, 64)).reshape((64, 64, channel))

# División del dataset en entrenamiento y validación (test)
x_train, x_val, y_train, y_val = train_test_split(real, labels, test_size=0.3, shuffle=True, random_state=42)


n_classes = len(np.unique(y_train))
# encoder = load_model('bagan_encoder.h5')
encoder = load_model('encoder_tennis4.h5', compile=True)
embedding = load_model('embedding_tennis4.h5')
encoder.compile(optimizer='adam', loss='mse')

# %% --------------------------------------- TSNE Visualization --------------------------------------------------------
def tsne_plot(encoder):
    "Creates and TSNE model and plots it"
    plt.figure(figsize=(8, 8))
    color = plt.get_cmap('tab10')

    latent = encoder.predict(x_train)  # with Encoder
    #latent = embedding.predict([latent, y_train])  ## with Embedding model
    tsne_model = TSNE(n_components=2, init='random', random_state=0)
    new_values = tsne_model.fit_transform(latent)
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    x = np.array(x)
    y = np.array(y)

    for c in range(n_classes):
        plt.scatter(x[y_train==c], y[y_train==c], c=np.array([color(c)]), label='%d' % c)
    plt.legend()
    plt.show()

tsne_plot(encoder)