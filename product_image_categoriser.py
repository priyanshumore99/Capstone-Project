import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os
import shutil
from PIL import Image

# Part 2 – Product Image Classifier (25 marks)

# 1. Load the dataset. Use Fashion-MNIST (Zalando Research), the free, keyless, automatically-downloadable benchmark of 70,000 grayscale 28x28 product images across 10 categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot -- an exact match for Flipkart's own apparel/footwear/accessories catalog. Pinned source: https://github.com/zalandoresearch/fashion-mnist (also fetchable with zero configuration via torchvision.datasets.FashionMNIST(root=..., download=True), which pulls from this same canonical dataset). Use the standard 60,000-image train split and 10,000-image test split; carve a stratified validation split (at least 5,000 images) out of the training set for model selection, leaving the test split untouched until final evaluation.

# Load dataset (60k train, 10k test)
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

print("Train full shape:", X_train_full.shape)
print("Test shape:", X_test.shape)

# Split off 5k validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=5000, stratify=y_train_full, random_state=42
)

print("Train shape:", X_train.shape)
print("Validation shape:", X_val.shape)
print("Test shape:", X_test.shape)

# Scale to [0,1]
X_train = X_train.astype("float32") / 255.0
X_val   = X_val.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

# Add channel dimension (grayscale → 1 channel)
X_train = X_train[..., tf.newaxis]
X_val   = X_val[..., tf.newaxis]
X_test  = X_test[..., tf.newaxis]

print("Train shape after channel add:", X_train.shape)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 2. Preprocess for a pretrained backbone. Replicate the single grayscale channel to 3 channels, resize to whatever input size your chosen pretrained backbone expects (document the exact size you use), and normalize with the ImageNet mean/std the backbone was originally trained with.

# 2.1. Load dataset (keep as 2D images: shape 28x28)
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Split off 5k validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=5000, stratify=y_train_full, random_state=42
)

# 2.2. Preprocess function: handles channel expansion, RGB conversion, and resizing per sample
def preprocess(img, label):
    img = tf.expand_dims(img, axis=-1)           # (28, 28, 1)
    img = tf.image.grayscale_to_rgb(img)         # (28, 28, 3)
    img = tf.image.resize(img, [224, 224])       # (224, 224, 3)
    img = tf.cast(img, tf.float32)              # EfficientNet handles internal normalization [0, 255]
    return img, label

# 2.3. Dataset pipelines
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(10000)
            .batch(64)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(64)
          .prefetch(tf.data.AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
           .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
           .batch(64)
           .prefetch(tf.data.AUTOTUNE))

# Sanity check output
for images, labels in train_ds.take(1):
    print("Batch images shape:", images.shape)  # Correct shape: (64, 224, 224, 3)

# 2.4. Build transfer learning model
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 2.5. Train & Evaluate
history = model.fit(train_ds, validation_data=val_ds, epochs=5)
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 3. Build the transfer-learning model. Load a pretrained CNN backbone (ResNet-18 or EfficientNet-B0), freeze the early and middle layers, and attach a new classifier head sized for 10 output classes. Train only the new head first (feature extraction). Document your batch size, optimizer (use Adam), learning rate, and number of epochs. Speed tip (do this, it changes the runtime by roughly an order of magnitude): since the backbone is frozen during feature extraction, run it once over every image to extract and cache its output features, then train only the small head on those cached feature vectors -- this is mathematically identical to re-running the frozen backbone's forward pass every epoch, but turns an hours-long CPU training loop into one dominated by a single feature-extraction pass (a few minutes on a GPU, and still well under an hour on a laptop CPU) followed by a near-instant head-only training step. If you have access to a free GPU runtime (e.g. Google Colab or Kaggle), that is also a fine alternative to caching, but caching alone is sufficient on a CPU-only machine.

# 3.1. Load Pretrained Backbone
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze all backbone layers

# Add global pooling to produce a 1D feature vector per image
feature_extractor = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D()
])

# 3.2. Speed Optimization: Feature Caching
# Extract feature vectors once to bypass the backbone forward pass during training
def extract_and_cache_features(dataset):
    features_list = []
    labels_list = []
    for images, labels in dataset:
        feats = feature_extractor(images, training=False)
        features_list.append(feats.numpy())
        labels_list.append(labels.numpy())
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

X_train_cached, y_train_cached = extract_and_cache_features(train_ds)
X_val_cached, y_val_cached = extract_and_cache_features(val_ds)

# 3.3. Build & Train Classification Head on Cached Features
head_model = models.Sequential([
    layers.Input(shape=(X_train_cached.shape[1],)),  # Input shape matches extracted feature dim (1280)
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(10, activation="softmax")
])

head_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = head_model.fit(
    X_train_cached, y_train_cached,
    validation_data=(X_val_cached, y_val_cached),
    epochs=10,
    batch_size=64
)

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 4. Fine-tune if needed. If your feature-extraction validation accuracy is below 80%, unfreeze the backbone's late layers (keep early/middle layers frozen) and continue training at a lower learning rate, using the standard gradual-unfreezing fine-tuning strategy. Document the before/after validation accuracy either way.

# Explaination
# Transfer Learning Model Evaluation & Fine-Tuning Decision
# Backbone Architecture: EfficientNetB0 (Pre-trained on ImageNet)
# Training Strategy: Feature Caching + Classification Head Training

# Key Metrics
# Final Feature-Extraction Val Accuracy: 93.12%
# Final Feature-Extraction Val Loss: 0.1957
# Fine-Tuning Threshold: 80.00%

# Unfreezing Decision & Summary
# Decision: Fine-Tuning Skipped (Not Required)
# The feature-extraction stage achieved a 93.12% validation accuracy, which comfortably exceeds the target threshold of 80.00%.
# Per the experimental protocol, the backbone layers remain frozen, and fine-tuning at a lower learning rate was not triggered. The cached feature classification head successfully generalized to the dataset without needing fine-tuning of the backbone's late layers.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#5. Evaluate. Report final test-set accuracy, a full 10x10 confusion matrix, and per-class precision/recall.


# 5.1. Extract and Cache Test Features (Speed Optimization)
print("Extracting test features...")
X_test_cached, y_test_cached = extract_and_cache_features(test_ds)

# 5.2. Generate Predictions using head_model
y_pred_probs = head_model.predict(X_test_cached, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = y_test_cached

# 5.3. Report Final Test-Set Loss & Accuracy
test_loss, test_acc = head_model.evaluate(X_test_cached, y_test_cached, verbose=0)

print("\n" + "=" * 55)
print(f" FINAL TEST-SET ACCURACY: {test_acc * 100:.2f}%")
print(f" FINAL TEST-SET LOSS:     {test_loss:.4f}")
print("=" * 55 + "\n")

# 5.4. Full 10x10 Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("--- 10x10 Confusion Matrix ---")
print(cm)
print()

# 5.5. Per-Class Precision & Recall
fashion_mnist_classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

print("--- Per-Class Precision, Recall, and F1-Score ---")
print(classification_report(y_true, y_pred, target_names=fashion_mnist_classes, digits=4))


#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 6. Document confusion patterns. Name at least two specific category pairs your model confuses most often (read them directly off your own confusion matrix -- do not guess), and write one paragraph per pair explaining, in terms of the actual visual similarity between those two apparel/footwear silhouettes, why the confusion is plausible.

# Mask diagonal (correct predictions) to find off-diagonal confusion pairs
cm_off_diag = cm.copy()
np.fill_diagonal(cm_off_diag, 0)

# Get top 2 misclassified pairs (True Class -> Predicted Class)
top_pairs = np.unravel_index(
    np.argsort(cm_off_diag.ravel())[-2:][::-1], cm_off_diag.shape
)

print("--- TOP CONFUSION PAIRS ---")
for i in range(2):
    true_idx, pred_idx = top_pairs[0][i], top_pairs[1][i]
    count = cm_off_diag[true_idx, pred_idx]
    print(
        f"{i+1}. True: '{fashion_mnist_classes[true_idx]}' --> Predicted: '{fashion_mnist_classes[pred_idx]}' ({count} misclassifications)"
    )
    
# Explaination of above Code
# 1. True: Shirt $\rightarrow$ Predicted: T-shirt/top (89 misclassifications)
#  Visual & Structural Analysis:
# Confusion between Shirt and T-shirt/top is the single largest error source for the model because both garments share an almost identical macro-silhouette—a short-sleeved torso outline with a central crew or V-neck profile. In the original $28 \times 28$ grayscale images, fine-grained discriminative details such as row buttons, collar stiffening, plackets, and sleeve seams are severely degraded or lost entirely during resizing. When processed by global average pooling, the model relies on broader geometric shapes, causing casual button-down shirts to closely mirror the spatial activation map of basic T-shirts.

# 2. True: Shirt $\rightarrow$ Predicted: shirt (81 misclassifications)
#  Visual & Structural Analysis:
# The secondary confusion between Shirt and Coat stems from structural similarities in long-sleeved, open-front upper garments with collar structures. Long-sleeved dress shirts or light overshirts share a similar aspect ratio, shoulder taper, and lapel/collar boundary with light jackets or coats. Without color or distinct fabric texture cues (e.g., heavy wool weave vs. thin cotton knit), the model struggles to differentiate the thickness or intent of the outer garment, leading to misclassification based purely on shared upper-body boundary contours.onfusion between Shirt and Coat stems from structural similarities in long-sleeved, open-front upper garments with collar structures. Long-sleeved dress shirts or light overshirts share a similar aspect ratio, shoulder taper, and lapel/collar boundary with light jackets or coats. Without color or distinct fabric texture cues (e.g., heavy wool weave vs. thin cotton knit), the model struggles to differentiate the thickness or intent of the outer garment, leading to misclassification based purely on shared upper-body boundary contours.

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 7. Save the artifact. Persist the trained model's weights (torch.save(model.state_dict(), ...) or the Keras/TensorFlow equivalent) to models/product_classifier.pt, plus a short, documented one-function loading + single-image-prediction snippet. This is what Part 3's classify_product_image tool will call.

# 7.1. Stitch Pretrained Backbone + Trained Head into End-to-End full_model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = head_model(x)

full_model = tf.keras.Model(inputs, outputs)

# 7.2. Persist Model Artifacts (Handles Keras 3 Strict Extensions)
os.makedirs("models", exist_ok=True)

# Keras 3 requires .weights.h5 or .keras
keras_weights_path = "models/product_classifier.weights.h5"
full_model_path = "models/product_classifier.keras"
target_pt_path = "models/product_classifier.pt"

# Save full model & weights
full_model.save(full_model_path)
full_model.save_weights(keras_weights_path)

# Copy/Alias to 'models/product_classifier.pt' to satisfy the exact requirement
shutil.copy(full_model_path, target_pt_path)

print(f"✅ Full model saved to '{full_model_path}'")
print(f"✅ Artifact persisted to required path '{target_pt_path}'")


# 7.3. Documented Single-Function Prediction Routine (Part 3 Tool interface)
def classify_product_image(image_input, model_path="models/product_classifier.pt"):
    """
    Loads the persisted product classifier model and predicts the category of a single image.
    
    Args:
        image_input (str or np.ndarray): File path to an image file or a 
                                         numpy array representing an image (shape: 28x28, 28x28x1, or 224x224x3).
        model_path (str): Path to saved model artifact. Defaults to 'models/product_classifier.pt'.
        
    Returns:
        dict: Containing 'class_id', 'class_name', and 'confidence' (float).
    """
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    # Safely handle Keras 3 format loading from .pt copy or .keras fallback
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception:
        # Fallback to .keras if Keras strict extension check triggers
        fallback_path = model_path.replace(".pt", ".keras")
        model = tf.keras.models.load_model(fallback_path)
        
    # Process input: Handle string file path vs numpy array
    if isinstance(image_input, str):
        img = tf.io.read_file(image_input)
        img = tf.image.decode_image(img, channels=1, expand_animations=False)
    else:
        img = tf.convert_to_tensor(image_input, dtype=tf.float32)
        if len(img.shape) == 2:
            img = tf.expand_dims(img, axis=-1)  # (28, 28) -> (28, 28, 1)

    # Standardize pipeline preprocessing
    if img.shape[-1] == 1:
        img = tf.image.grayscale_to_rgb(img)    # (28, 28, 1) -> (28, 28, 3)
    img = tf.image.resize(img, [224, 224])      # Resize to model input shape
    img = tf.cast(img, tf.float32)
    
    # Add batch dimension: (1, 224, 224, 3)
    img_batch = tf.expand_dims(img, axis=0)
    
    # Run inference
    probs = model.predict(img_batch, verbose=0)[0]
    pred_class_id = int(np.argmax(probs))
    
    return {
        "class_id": pred_class_id,
        "class_name": class_names[pred_class_id],
        "confidence": float(probs[pred_class_id])
    }

# Sanity Test: Calling with default parameter
print(classify_product_image(X_test[0]))

# 7.4. Verification Test
sample_result = classify_product_image(X_test[0])
print("\n--- Test Prediction Result ---")
print(f"Predicted Class : {sample_result['class_name']}")
print(f"Class ID        : {sample_result['class_id']}")
print(f"Confidence      : {sample_result['confidence'] * 100:.2f}%")

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 8. Export real sample images as actual image files. torchvision.datasets.FashionMNIST stores its data as raw IDX binary files, not a folder of individual images -- there is no ready-made image file for Part 3's classify_product_image(image_path: str) tool to point at. Pick at least 5 real test-split images (ideally covering different classes) and write each one out as an actual .png file via PIL.Image.fromarray(...) (or the equivalent in your chosen framework) to data/sample_images/, named so the true label is obvious from the filename (e.g. data/sample_images/03_sneaker.png). Commit these exact .png files to your repo -- this is what Part 3's tool will be pointed at, not the raw IDX data.

# 8.1. Define output directory
output_dir = "data/sample_images/"
os.makedirs(output_dir, exist_ok=True)

# 8.2. Fashion-MNIST class labels
fashion_mnist_classes = [
    "t_shirt_top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
]

# 8.3. Select 5 distinct classes from test set
selected_indices = []
seen_classes = set()

for idx, label in enumerate(y_test):
    if label not in seen_classes:
        seen_classes.add(label)
        selected_indices.append(idx)
    if len(selected_indices) == 5:
        break

# 8.4. Save PNG files
exported_files = []
for idx in selected_indices:
    label_id = int(y_test[idx])
    label_name = fashion_mnist_classes[label_id]

    # Retrieve 28x28 raw uint8 image
    # Note: If X_test is normalized [0,1], scale back to [0, 255] uint8
    img_array = X_test[idx]
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    # Remove channel dimension if shape is (28, 28, 1)
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)

    # Create PIL image and save
    img = Image.fromarray(img_array)
    filename = f"{output_dir}{label_id:02d}_{label_name}.png"
    img.save(filename)
    exported_files.append((filename, label_name))

print("✅ Sample PNG images exported successfully:\n")
for filepath, label_name in exported_files:
    print(f" - {filepath} (True Label: {label_name})")
    
# Test image prediction on one of the exported file paths
sample_path = exported_files[0][0]  # e.g., 'data/sample_images/00_t_shirt_top.png'

result = classify_product_image(sample_path)

print(f"\n--- Testing classify_product_image with File Path ---")
print(f"Input File      : {sample_path}")
print(f"Predicted Class : {result['class_name']}")
print(f"Confidence      : {result['confidence'] * 100:.2f}%")