import tensorflow as tf
import os
from rasa import train

# Configure TensorFlow for Apple Silicon
tf.config.set_visible_devices([], 'GPU')  # Disables GPU fallback
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Forces legacy optimizers

def main():
    # Training configuration
    training_result = train(
        domain="domain.yml",
        config="config.yml",
        training_files=["data/"],
        output="models/"
    )
    print(f"Model trained and saved at: {training_result.model}")

if __name__ == "__main__":
    main()