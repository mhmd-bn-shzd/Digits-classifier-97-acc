# Digits Classifier on MNIST dataset

This model is a ResNet inspired handwritten decimal digit classifier built using TensorFlow Keras Functional API.

## Contents:

- `ResNet.ipynb` for model experimentation and training
- `predict.py` for running inference on the model
- Pretrained model files: `Classifier.weights.h5`, `DigitClassifier.h5`, and `ResNetDigitClassifier.keras`
- A sample image used by the prediction script

## Requirements

- Python 3.10+
- TensorFlow
- OpenCV
- NumPy

## Usage

Run the prediction script to classify the sample digit image:

```bash
python predict.py
```

The script loads `Classifier.weights.h5`, preprocesses `seven.png`, and prints the predicted digit with confidence.

## Notes

- The model expects grayscale images resized to `28 x 28`.
- You can swap `seven.png` with another digit image as long as it follows the same format.
