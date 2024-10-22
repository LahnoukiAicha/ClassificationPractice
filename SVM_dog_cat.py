import os
import numpy as np
import cv2
import pickle
"""
# Directory containing the images
dir ='C:\\Users\\Aicha LAHNOUKI\\Desktop\\data project\\classification\\PetImages'
categories = ['Cat', 'Dog']
data = []

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            pet_img = cv2.imread(img_path, 0) # open in greyscale mode
            pet_img = cv2.resize(pet_img, (50, 50))
            image = np.array(pet_img).flatten()
            data.append([image, label])
        except Exception as e:
            print(f"Failed to process image {img_path}: {e}")

# Check if data is collected
if len(data) > 0:
    with open('data1.pickle', 'wb') as pick_out:
        pickle.dump(data, pick_out)
else:
    print("No data was collected. Please check the images and paths.")
"""

import pickle

# Load the data
try:
    with open('data1.pickle', 'rb') as pick_in:
        data = pickle.load(pick_in)
except (EOFError, FileNotFoundError):
    print("Error: The file is empty, corrupted, or missing.")
    data = None

# Proceed only if data was successfully loaded
if data:
    features = []
    labels = []  # class that represents the data (0=cat, 1=dog)
    for feature, label in data:
        features.append(feature)
        labels.append(label)

    # Split the data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.01)

    # Train the model
    from sklearn.svm import SVC
    svm_model = SVC(C=1, gamma='auto', kernel='poly')
    svm_model.fit(X_train, y_train)

    # Save the trained model
    with open('model.sav', 'wb') as saved_model:
        pickle.dump(svm_model, saved_model)

    # Evaluate the model
    accuracy = svm_model.score(X_test, y_test)
    print('Accuracy:', accuracy)
else:
    print("Data loading failed. Please ensure the data is correctly collected and saved.")
