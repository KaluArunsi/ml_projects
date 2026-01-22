"""
Model evaluation and visualization
"""
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def evaluate_model(model, feature_test, label_test, verbose=True):
    """
    Evaluate model performance on test set
    
    Returns MAE for use in prediction intervals
    """
    predictions = model.predict(feature_test).flatten()
    r2 = r2_score(label_test, predictions)
    
    # Get MAE from last training metrics
    mae = model.evaluate(feature_test, label_test, verbose=0)[2]
    
    if verbose:
        print(f"\nModel Performance:")
        print(f"R² Score: {r2:.4f} ({r2*100:.2f}%)")
        print(f"Mean Absolute Error: {mae:.2f} bikes")
    
    return mae


def plot_training_history(history):
    """
    Visualize training and validation loss over epochs
    """
    plt.figure(figsize=(8, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlim([0, 400])
    plt.ylim([0, 200])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.title('Model Training History')
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("\nTraining history saved to training_history.png")