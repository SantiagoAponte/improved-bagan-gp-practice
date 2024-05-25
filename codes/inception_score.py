import numpy as np
from numpy import asarray
from numpy.random import normal
from skimage.transform import resize
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.stats import entropy

# Prepare the inception v3 model
model = InceptionV3(include_top=True, pooling='avg', input_shape=(299, 299, 3))

# Function to scale images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # Resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0) * 255
        # Store
        images_list.append(new_image)
    return asarray(images_list)

# Function to calculate inception score
def calculate_inception_score(images, model, splits=8):
    # Preprocess images
    images = preprocess_input(images)
    # Predict class probabilities
    preds = model.predict(images)
    # Calculate the inception score
    scores = []
    n_part = images.shape[0] // splits
    for i in range(splits):
        part = preds[i * n_part:(i + 1) * n_part, :]
        py = np.mean(part, axis=0)
        scores.append(np.exp(np.mean([entropy(pyx, py) for pyx in part])))
    return np.mean(scores), np.std(scores)

# Load generator
gen_path = 'bagan_gp_tennis5_epoch6.h5'
generator = load_model(gen_path)

# Load your dataset
train_data = np.load('dataset.npz')
images = train_data['x']
labels = train_data['y']

# Split the dataset into training and validation (test)
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.3, shuffle=True, random_state=38)

# Assign real_imgs and real_label to the validation images and labels
real_imgs = x_val
real_label = y_val

# Calculate the number of images available per class in the validation set
#class_counts = {c: np.sum(real_label == c) for c in np.unique(real_label)}

# Select the sample size based on the minimum number of images available per class
sample_size = 50 #min(class_counts.values())

#print(f"Sample size set to {sample_size} based on available images per class")

# Calculate Inception Score for each class
n_classes = len(np.unique(real_label))
for c in range(n_classes):
    ########### Get generated samples by class ###########
    label = np.ones(sample_size) * c
    noise = normal(0, 1, (sample_size, generator.input_shape[0][1]))
    print('Latent dimension:', generator.input_shape[0][1])
    gen_samples = generator.predict([noise, label])
    gen_samples = gen_samples * 0.5 + 0.5

    ########### Get real samples by class ###########
    real_samples = real_imgs[real_label == c]
    real_samples = real_samples.astype('float32') / 255.

    # Resize images
    gen_samples = scale_images(gen_samples, (299, 299, 3))
    real_samples = scale_images(real_samples, (299, 299, 3))
    print('Scaled', gen_samples.shape, real_samples.shape)

    # Preprocess images
    gen_samples = preprocess_input(gen_samples)

    # Calculate Inception Score
    is_mean, is_std = calculate_inception_score(gen_samples, model)
    print('>>Inception Score(%d): Mean=%.3f, Std=%.3f' % (c, is_mean, is_std))
    print('-' * 50)
