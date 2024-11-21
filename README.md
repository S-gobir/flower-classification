# Flower Classification Model

This repository contains a flower classification deep learning model that classifies images into 14 flower categories. The model was trained using a dataset sourced from Kaggle.

## Dataset

The dataset consists of:
- **Training set**: 13,642 images belonging to 14 classes.
- **Validation set**: 98 images belonging to 14 classes.

### Classes
The dataset contains the following flower categories with respective item counts:
- **astilbe**: 726 items
- **bellflower**: 872 items
- **black-eyed susan**: 986 items
- **calendula**: 1,011 items
- **california poppy**: 1,021 items
- **carnation**: 924 items
- **common daisy**: 978 items
- **coreopsis**: 1,035 items
- **dandelion**: 1,038 items
- **iris**: 1,041 items
- **rose**: 986 items
- **sunflower**: 1,013 items
- **tulip**: 1,034 items
- **water lily**: 977 items

The dataset is sourced from Kaggle: [Flower Classification Dataset](https://www.kaggle.com/datasets/marquis03/flower-classification/data).

## Training Results

The model was trained over 5 epochs using a validation set for performance evaluation. Below are the results for each epoch:

| Epoch | Accuracy | Loss  | Validation Accuracy | Validation Loss |
|-------|----------|-------|---------------------|-----------------|
| 1     | 66.08%   | 1.0822 | 82.65%              | 0.4873          |
| 2     | 84.75%   | 0.4622 | 89.80%              | 0.3377          |
| 3     | 87.42%   | 0.3870 | 83.67%              | 0.4427          |
| 4     | 89.31%   | 0.3292 | 83.67%              | 0.4353          |
| 5     | 89.48%   | 0.3195 | 80.61%              | 0.5411          |

The best validation accuracy achieved during training was **89.80%** in epoch 2.

## Notebook

The training and evaluation processes are documented in the Jupyter Notebook file: `flower classification.ipynb`. The notebook includes:
1. Data loading and preprocessing.
2. Model architecture and compilation.
3. Training and evaluation results.
4. Visualization of predictions.

## Installation

To run this notebook locally, ensure you have the following prerequisites installed:
- Python 3.7+
- TensorFlow 2.x
- Matplotlib
- NumPy
- Pandas

Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository:
   ```bash
   git clone <https://github.com/S-gobir/flower-classification>
   cd <FLOWER CLASSIFICATION >
   ```
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/marquis03/flower-classification/data) and place it in the `data` directory.
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook "flower classification.ipynb"
   ```

## Model Prediction

The model predicts the category of a flower based on the provided image. Use the following code snippet to load the model and make predictions:

```python
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model("flower_model.h5")
class_labels = ["astilbe", "bellflower", "black_eyed_susan", "calendula",
                "california_poppy", "carnation", "common_daisy", "coreopsis",
                "dandelion", "iris", "rose", "sunflower", "tulip", "water_lily"]

def predict_flower(image_path):
    image = Image.open(image_path).resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    predictions = model.predict(image_array)
    predicted_label = class_labels[np.argmax(predictions)]
    return predicted_label

# Example usage
print(predict_flower("path/to/image.jpg"))
```

## Results Visualization

Predicted images with their labels can be visualized using the notebook. Predictions include confidence scores for each class.

## License

This project is licensed under the MIT License.
```

