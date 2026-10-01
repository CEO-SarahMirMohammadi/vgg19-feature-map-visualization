from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model

import matplotlib.pyplot as plt
from numpy import expand_dims


def plot_feature_maps(feature_maps, layer_number):
    feature_maps = feature_maps[0]

    # Display up to 16 feature maps
    num_maps = min(feature_maps.shape[-1], 16)

    rows = 4
    cols = 4

    plt.figure(figsize=(12, 12))

    for i in range(num_maps):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_maps[:, :, i], cmap="gray")
        plt.axis("off")
        plt.title(f"Map {i + 1}")

    plt.suptitle(
        f"VGG19 Feature Maps - Convolutional Layer {layer_number}"
    )

    plt.tight_layout()
    plt.show()


# Load the VGG19 model
base_model = VGG19(weights="imagenet")

# Select the hidden layer to visualize
# Try n = 2, 3, and 4
n = 2

# Redefine model to output the selected hidden layer
model = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[n].output
)

print(f"\nSelected layer: {n}")
print(f"Layer name: {base_model.layers[n].name}")

model.summary()

# Load the image with the required shape
img = load_img(
    "Elephant.jpg",
    target_size=(224, 224)
)

# Convert the image to an array
img = img_to_array(img)

# Expand dimensions to represent a single sample
img = expand_dims(img, axis=0)

# Prepare the image for VGG19
img = preprocess_input(img)

# Get feature maps
feature_maps = model.predict(img, verbose=0)

print("\nFeature maps shape:")
print(feature_maps.shape)

# Plot the feature maps
plot_feature_maps(feature_maps, n)