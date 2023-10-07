from tensorflow import keras
from keras import layers, callbacks
import pandas as pd
import matplotlib.pyplot as plt

# Read input data and split into training and validation data
red_wine = pd.read_csv('red-wine.csv')
df_train = red_wine.sample(frac=0.7, random_state=0)
df_valid = red_wine.drop(df_train.index)
print(df_train.head())

# Normalize the features to [0, 1] interval
max_ = df_train.max(axis=0)
min_ = df_train.min(axis=0)
df_train = (df_train - min_) / (max_ - min_)
df_valid = (df_valid - min_) / (max_ - min_)

# Split features and target
X_train = df_train.drop('quality', axis=1)
X_valid = df_valid.drop('quality', axis=1)
y_train = df_train['quality']
y_valid = df_valid['quality']

# Implement early stopping of the model to prevent overfitting
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,  # minimum amount of change to count as improvement
    patience=20,  # how many epochs to wait before stopping
    restore_best_weights=True,
)

# Create a network
model = keras.Sequential([
    layers.Dense(units=1024, activation='relu', input_shape=[11]),
    layers.Dropout(rate=0.3),
    layers.BatchNormalization(),
    layers.Dense(units=1024, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.BatchNormalization(),
    layers.Dense(units=1024, activation='relu'),
    layers.Dropout(rate=0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])

# Compile the model with the optimizer and loss function
model.compile(
    optimizer='adam',
    loss='mae',
)

# Train the model and store a History object
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=50,
    # callbacks=[early_stopping],
    verbose=False,
)

# Plot the loss during training
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print(f"Minimum validation loss: {history_df['val_loss'].min()}")
plt.show()
