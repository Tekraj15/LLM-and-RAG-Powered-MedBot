import tensorflow as tf
import os
from rasa import train

# Configure TensorFlow for Apple Silicon (M1/M2)
tf.config.set_visible_devices([], 'GPU')  # Disables GPU fallback
os.environ['TF_USE_LEGACY_KERAS'] = '1'   # Forces legacy optimizers

def main():
    # Training configuration
    # "rasa/" prefix to all paths so it finds the files correctly
    training_result = train(
        domain="rasa/domain.yml",
        config="rasa/config.yml",
        training_files=["rasa/data/"],
        output="rasa/models/"
    )
    print(f"Model trained and saved at: {training_result.model}")

if __name__ == "__main__":
    main()