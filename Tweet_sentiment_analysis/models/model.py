from tensorflow.keras.models import Sequential
from tensorflow as tf

def build_model():
    model=Sequential([
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.LayerNormalization(epsilon=1e-6),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(64,activation='relu'),
        tf.keras.layers.Dense(1,activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
