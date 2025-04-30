#pip install tensorflow
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


file_path = '/Users/willweiwu/Library/CloudStorage/Dropbox/CU Boulder/2025 Spring/INFO 5653-001 Text Mining/Project/wiki/wiki_data_nb_dt_svm.csv'
df = pd.read_csv(file_path)

display(df.head())

X = df.drop('sentiment', axis=1).values
y_raw = df['sentiment'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw) 
y = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_sample = pd.DataFrame(X_train[:5], columns=df.columns[:-1])
train_sample['sentiment'] = label_encoder.inverse_transform(np.argmax(y_train[:5], axis=1))

test_sample = pd.DataFrame(X_test[:5], columns=df.columns[:-1])
test_sample['sentiment'] = label_encoder.inverse_transform(np.argmax(y_test[:5], axis=1))

#sample: training data
display(train_sample)

#sample: test data
display(test_sample)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)


y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)


acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {acc:.4f}")


cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()
