"""
Neural network model architecture and training
"""
import keras


def build_model(n_features):
    """
    Build feedforward neural network for bike demand prediction
    
    Architecture favors stability and interpretability over complexity
    Uses MAE loss which aligns with business impact (prediction error in bike units)
    """
    model = keras.Sequential([
        keras.layers.Dense(32, activation='gelu', input_shape=(n_features,)),
        keras.layers.Dense(16, activation='gelu'),
        keras.layers.Dense(1)
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=0.002)
    model.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['mse', 'mae'])
    
    return model


def train_model(model, feature_train, label_train, epochs=500, batch_size=1024, verbose=True):
    """
    Train model with early stopping to prevent overfitting
    
    Returns training history for analysis
    """
    callbacks = [keras.callbacks.EarlyStopping(patience=40)]
    
    history = model.fit(
        feature_train, label_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1 if verbose else 0,
        shuffle=False,
        callbacks=callbacks
    )
    
    return history