"""
=============================================================
  Kyungdong University Global - Smart Computing (SC)
  Problem Solving Concepts - Final Project (Project #05)

  Title : Fast VGG-Style CNN for MNIST Digit Classification
  Tool  : Python + OpenCV + TensorFlow/Keras
=============================================================

"""

# ─────────────────────────────────────────────────────────────
# 1. IMPORTS
# ─────────────────────────────────────────────────────────────

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Enable TensorFlow optimization
tf.config.optimizer.set_jit(True)

print("TensorFlow Version :", tf.__version__)
print("OpenCV Version     :", cv2.__version__)


# ─────────────────────────────────────────────────────────────
# 2. LOAD DATASET
# ─────────────────────────────────────────────────────────────

print("\n[1/5] Loading MNIST dataset...")

(X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()

# Combine datasets for custom split
X_all = np.concatenate([X_train_full, X_test_full], axis=0)
y_all = np.concatenate([y_train_full, y_test_full], axis=0)

print("Total Samples:", len(X_all))


# ─────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────

print("\n[2/5] Preprocessing images...")

TARGET_SIZE = 32

def preprocess_images(images):

    processed = []

    for img in images:

        # Resize image
        resized = cv2.resize(
            img,
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=cv2.INTER_AREA
        )

        # Normalize
        resized = resized.astype(np.float32) / 255.0

        # Add channel dimension
        resized = np.expand_dims(resized, axis=-1)

        processed.append(resized)

    return np.array(processed)


X_processed = preprocess_images(X_all)

print("Processed Shape:", X_processed.shape)

# One-hot encoding
y_encoded = to_categorical(y_all, num_classes=10)


# ─────────────────────────────────────────────────────────────
# 4. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────

print("\n[3/5] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_all
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# ─────────────────────────────────────────────────────────────
# 5. BUILD FAST VGG-STYLE MODEL
# ─────────────────────────────────────────────────────────────

print("\n[4/5] Building VGG-style CNN model...")

def build_vgg_style_model(input_shape=(32, 32, 1), num_classes=10):

    model = models.Sequential(name="Fast_VGG_Style_MNIST")

    # ── Block 1 ─────────────────────────────
    model.add(layers.Conv2D(
        32,
        (3,3),
        padding='same',
        activation='relu',
        input_shape=input_shape
    ))

    model.add(layers.Conv2D(
        32,
        (3,3),
        padding='same',
        activation='relu'
    ))

    model.add(layers.MaxPooling2D((2,2)))

    # ── Block 2 ─────────────────────────────
    model.add(layers.Conv2D(
        64,
        (3,3),
        padding='same',
        activation='relu'
    ))

    model.add(layers.Conv2D(
        64,
        (3,3),
        padding='same',
        activation='relu'
    ))

    model.add(layers.MaxPooling2D((2,2)))

    # ── Block 3 ─────────────────────────────
    model.add(layers.Conv2D(
        128,
        (3,3),
        padding='same',
        activation='relu'
    ))

    model.add(layers.Conv2D(
        128,
        (3,3),
        padding='same',
        activation='relu'
    ))

    model.add(layers.MaxPooling2D((2,2)))

    # ── Global Pooling ──────────────────────
    model.add(layers.GlobalAveragePooling2D())

    # ── Dense Head ──────────────────────────
    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


model = build_vgg_style_model()

model.summary()


# ─────────────────────────────────────────────────────────────
# 6. COMPILE MODEL
# ─────────────────────────────────────────────────────────────

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [

    EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )
]


# ─────────────────────────────────────────────────────────────
# 7. TRAIN MODEL
# ─────────────────────────────────────────────────────────────

print("\n[5/5] Training model...")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128,
    callbacks=callbacks,
    verbose=1
)


# ─────────────────────────────────────────────────────────────
# 8. EVALUATION
# ─────────────────────────────────────────────────────────────

print("\n─── EVALUATION ─────────────────────────")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy : {test_acc * 100:.2f}%")
print(f"Test Loss     : {test_loss:.4f}")


# Predictions
y_pred_probs = model.predict(X_test)

y_pred = np.argmax(y_pred_probs, axis=1)

y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")

print(classification_report(
    y_true,
    y_pred,
    target_names=[str(i) for i in range(10)]
))


# ─────────────────────────────────────────────────────────────
# 9. VISUALIZATION
# ─────────────────────────────────────────────────────────────

# ── Training History ───────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14,5))

# Accuracy
axes[0].plot(history.history['accuracy'], label='Train Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')

axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

# Loss
axes[1].plot(history.history['loss'], label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')

axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

print("Saved: training_history.png")


# ── Confusion Matrix ───────────────────────

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=range(10),
    yticklabels=range(10)
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

print("Saved: confusion_matrix.png")


# ── Sample Predictions ─────────────────────

fig, axes = plt.subplots(4, 8, figsize=(16,8))

indices = np.random.choice(len(X_test), 32, replace=False)

for i, idx in enumerate(indices):

    ax = axes[i // 8][i % 8]

    img = X_test[idx].squeeze()

    ax.imshow(img, cmap='gray')

    color = 'green' if y_pred[idx] == y_true[idx] else 'red'

    ax.set_title(
        f"P:{y_pred[idx]} T:{y_true[idx]}",
        color=color,
        fontsize=8
    )

    ax.axis('off')

plt.tight_layout()

plt.savefig("sample_predictions.png")

plt.show()

print("Saved: sample_predictions.png")


# ─────────────────────────────────────────────────────────────
# 10. FINAL RESULT
# ─────────────────────────────────────────────────────────────

print("\n✅ PROJECT COMPLETE!")
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")