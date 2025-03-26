import tensorflow as tf
import pandas as pd
import os
from tensorflow.keras import layers, models #type:ignore
from tensorflow.keras.models import load_model #type:ignore

# Charger les labels d'entraînement
train_df = pd.read_csv('/Users/nikolavucic/Documents/CNN_Butterfly/archive/Training_set.csv')

# Extraire les chemins des images et les labels
train_images = ['/Users/nikolavucic/Documents/CNN_Butterfly/archive/train/' + fname for fname in train_df['filename']]
train_labels = train_df['label']

# Mapper les labels texte à des indices entiers
label_mapping = {label: idx for idx, label in enumerate(sorted(set(train_labels)))}
train_labels = [label_mapping[label] for label in train_labels]


# Fonction pour traiter les images
def process_image(filename, label):
    image = tf.io.read_file(filename)  # Lire l'image depuis son chemin
    image = tf.image.decode_jpeg(image, channels=3)  # Décoder l'image en RGB
    image = tf.image.resize(image, [224, 224])  # Redimensionner à 224x224
    image = image / 255.0  # Normaliser les pixels entre 0 et 1
    return image, label

# Créer le Dataset d'entraînement
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.map(process_image).shuffle(buffer_size=100).batch(32)


# Charger le fichier CSV des données de test
test_df = pd.read_csv('/Users/nikolavucic/Documents/CNN_Butterfly/archive/Testing_set.csv')
test_images = ['/Users/nikolavucic/Documents/CNN_Butterfly/archive/test/' + fname for fname in test_df['filename']]

# Créer le Dataset de test
test_dataset = tf.data.Dataset.from_tensor_slices(test_images)
test_dataset = test_dataset.map(lambda x: process_image(x, 0)).batch(32)

# Définir un modèle CNN simple
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(label_mapping), activation='softmax')  # Nombre de classes
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print(model.summary())

# Entraîner le modèle

print("\n=== Entraînement du modèle 1 ===")

model.fit(train_dataset, epochs=10)

predictions = model.predict(test_dataset)
predicted_labels = [list(label_mapping.keys())[idx] for idx in tf.argmax(predictions, axis=1).numpy()]
print(predicted_labels)

#4. Évaluation et amélioration du modèle
print("\n=== Évaluation sur les données de test ===")
loss, accuracy = model.evaluate(test_dataset)
print(f"Précision du modèle : {accuracy:.2f}")



# Charger le modèle sans ré-exécuter l'entraînement
#model = load_model("butterfly_model.h5")