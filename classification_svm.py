import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load the images and convert them to a dataframe
categories = ['cats', 'dogs']
flat_data_arr = []
target_arr = []
datadir = 'IMAGES/'

for i in categories:
    path = os.path.join(datadir, i)
    for img in os.listdir(path):
        img_array = imread(os.path.join(path, img))
        img_resized = resize(img_array, (150, 150, 3))
        flat_data_arr.append(img_resized.flatten())
        target_arr.append(categories.index(i))

# Convert lists to numpy arrays
flat_data = np.array(flat_data_arr)
target = np.array(target_arr)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(flat_data, target, test_size=0.3, random_state=109)

# Normalize the data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Train the SVM model
model = svm.SVC(kernel='linear', C=0.5)
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

img_array = imread('IMAGES/prob_cat.jpg')
img_resized = resize(img_array, (150, 150, 3))
flat_data_arr = np.array(img_resized.flatten())
flat_data = scaler.transform([flat_data_arr])
print(categories[model.predict(flat_data)[0]])


