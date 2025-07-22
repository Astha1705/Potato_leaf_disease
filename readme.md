# Plant Disease Classifier

This is a beginner-friendly image classification project built with TensorFlow. The model is trained on the PlantVillage dataset to classify plant diseases from leaf images.

## Folder Structure

```
plant_disease_classifier/
│
├── data/                    # Dataset folder
│   └── PlantVillage/       # Subfolders for each class
│
├── models/                 # Trained model will be saved here
│   └── 1/                  # Versioned model directory
│
├── src/                    # Source code
│   ├── train.py            # Script for training the model
│   ├── predict.py          # Script to predict on new image
│   └── utils.py            # Helper functions
│
├── sample_leaf.jpg         # Test image for prediction
├── requirements.txt        # List of dependencies
├── README.md               # Project documentation
```

## How to Use

### 1. Set Up Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
.\venv\Scripts\activate         # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python src/train.py
```

- This will train the CNN model using images in `data/PlantVillage/`
- After training, the model will be saved to `models/1/`
- A plot of training/validation accuracy and loss will be saved as `training_results.png`

### 4. Run Prediction

Make sure you have a test image named `sample_leaf.jpg` in the project root folder.

```bash
python src/predict.py
```

This will load the trained model and predict the class and confidence score of the input image.

## Requirements

- Python 3.8+
- TensorFlow
- matplotlib
- numpy

Install with:

```bash
pip install tensorflow matplotlib numpy
```

## Notes

- The number of classes is automatically detected based on the subfolders in `PlantVillage`
- The model uses image resizing, normalization, and simple data augmentation
- Modify `IMAGE_PATH` in `predict.py` to test on any custom image

## Credits

- Dataset: PlantVillage Dataset
- TensorFlow and Keras for deep learning



