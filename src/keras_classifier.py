import os
# Set TensorFlow as backend - required for TFSMLayer
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
from PIL import Image
from pathlib import Path

class KerasDigitClassifier:
    def __init__(self, model_path="models/digit_classifier_keras"):
        """
        Initialize the Keras digit classifier.
        
        Args:
            model_path: Path to the local model (relative to project root)
        """
        self.is_tfsm = False
        
        # Convert to absolute path if relative
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            # Get current working directory (project root)
            cwd = Path.cwd()
            # Try src directory first
            if (cwd / "src").exists():
                model_path_obj = cwd / "src" / model_path
            if not model_path_obj.exists():
                model_path_obj = cwd / model_path
        
        print(f"[+] Loading Keras model from: {model_path_obj}")
        
        # Check what files exist in the model directory
        if model_path_obj.is_dir():
            print(f"[+] Model directory contents:")
            for file in model_path_obj.rglob("*"):
                if file.is_file():
                    print(f"    - {file.relative_to(model_path_obj)}")
            
            # Check for SavedModel format
            if (model_path_obj / "saved_model.pb").exists() or (model_path_obj / "saved_model.pbtxt").exists():
                print(f"[+] Detected TensorFlow SavedModel format")
                self.is_tfsm = True
            # Check for .keras files
            elif list(model_path_obj.rglob("*.keras")):
                keras_file = list(model_path_obj.rglob("*.keras"))[0]
                print(f"[+] Found .keras file: {keras_file.name}")
                model_path_obj = keras_file
            # Check for .h5 files
            elif list(model_path_obj.rglob("*.h5")):
                h5_file = list(model_path_obj.rglob("*.h5"))[0]
                print(f"[+] Found .h5 file: {h5_file.name}")
                model_path_obj = h5_file
            else:
                # If directory has no recognizable files, assume SavedModel format
                print(f"[+] No .keras or .h5 files found, assuming SavedModel format")
                self.is_tfsm = True
        
        # Load the model based on format
        try:
            if self.is_tfsm:
                # TensorFlow SavedModel format - use TFSMLayer
                print(f"[+] Loading as TensorFlow SavedModel using TFSMLayer")
                self.model = keras.layers.TFSMLayer(str(model_path_obj), call_endpoint='serving_default')
            else:
                # Keras or H5 format
                print(f"[+] Loading as Keras model")
                self.model = keras.saving.load_model(str(model_path_obj))
            
            print(f"[+] Keras model loaded successfully")
        except Exception as e:
            print(f"[!] Error loading model: {e}")
            raise
        
    def preprocess_image(self, img):
        """
        Preprocess a PIL image for the Keras model.
        Adjust this based on your model's expected input format.
        """
        # Convert to grayscale (model expects single channel)
        if img.mode != 'L':
            img = img.convert('L')
        
        # Resize to expected input size (28x28 for MNIST-like models)
        img = img.resize((28, 28))
        
        # Convert to array and normalize
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Reshape to (28, 28, 1) - grayscale with channel dimension
        img_array = img_array.reshape(28, 28, 1)
        
        # Add batch dimension: (1, 28, 28, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, img):
        """
        Predict digit class probabilities for a single digit image.
        
        Args:
            img: PIL Image of a single digit
            
        Returns:
            numpy array of class probabilities (shape: [10])
        """
        preprocessed = self.preprocess_image(img)
        
        if self.is_tfsm:
            # TFSMLayer returns a dict with output names as keys
            predictions = self.model(preprocessed)
            # Extract the actual predictions from the dict
            # Common output names: 'output_0', 'dense', 'predictions', etc.
            if isinstance(predictions, dict):
                # Try common output names
                for key in ['output_0', 'predictions', 'dense', 'logits']:
                    if key in predictions:
                        predictions = predictions[key]
                        break
                else:
                    # Use the first value if no standard key found
                    predictions = list(predictions.values())[0]
            
            # Convert to numpy if needed
            predictions = np.array(predictions)
        else:
            predictions = self.model.predict(preprocessed, verbose=0)
        
        # Ensure we return a 1D array of probabilities
        if len(predictions.shape) > 1:
            predictions = predictions[0]
        
        # Apply softmax if the output doesn't sum to 1 (might be logits)
        if not np.isclose(predictions.sum(), 1.0, atol=0.01):
            exp_preds = np.exp(predictions - np.max(predictions))
            predictions = exp_preds / exp_preds.sum()
        
        return predictions
    
    def predict_top_n(self, img, n=2):
        """
        Get top N predictions for a digit image.
        
        Args:
            img: PIL Image of a single digit
            n: Number of top predictions to return
            
        Returns:
            List of tuples (class_label, probability) sorted by probability
        """
        probs = self.predict(img)
        # Get top N indices
        top_n_indices = np.argsort(probs)[::-1][:n]
        top_n_results = [(str(idx), float(probs[idx])) for idx in top_n_indices]
        return top_n_results
    
    def predict_batch(self, images):
        """
        Predict digit class probabilities for multiple images at once (batch inference).
        
        Args:
            images: List of PIL Images
            
        Returns:
            numpy array of shape (batch_size, 10) with probabilities for each image
        """
        if not images:
            return np.array([])
        
        # Preprocess all images
        batch = []
        for img in images:
            preprocessed = self.preprocess_image(img)
            batch.append(preprocessed[0])  # Remove batch dimension for stacking
        
        # Stack into single batch
        batch_array = np.stack(batch, axis=0)
        
        if self.is_tfsm:
            # TFSMLayer returns a dict with output names as keys
            predictions = self.model(batch_array)
            # Extract the actual predictions from the dict
            if isinstance(predictions, dict):
                for key in ['output_0', 'predictions', 'dense', 'logits']:
                    if key in predictions:
                        predictions = predictions[key]
                        break
                else:
                    predictions = list(predictions.values())[0]
            predictions = np.array(predictions)
        else:
            predictions = self.model.predict(batch_array, verbose=0)
        
        # Apply softmax if outputs don't sum to 1 (might be logits)
        for i in range(len(predictions)):
            if not np.isclose(predictions[i].sum(), 1.0, atol=0.01):
                exp_preds = np.exp(predictions[i] - np.max(predictions[i]))
                predictions[i] = exp_preds / exp_preds.sum()
        
        return predictions
