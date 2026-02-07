# CNN Cats and Dogs Classifier

A deep learning image classification project that uses a Convolutional Neural Network (CNN) to distinguish between cats and dogs. Built with TensorFlow and Keras, this model demonstrates fundamental computer vision techniques for binary image classification.

## 🎯 Project Overview

This project implements a CNN-based image classifier trained on a dataset of cat and dog images. The model achieves high accuracy by employing modern deep learning techniques including batch normalization, dropout regularization, and early stopping.

## 🏗️ Model Architecture

The CNN model consists of:

- **4 Convolutional Blocks**, each containing:
  - Conv2D layer (32 → 64 → 128 → 256 filters)
  - Batch Normalization
  - MaxPooling2D (2×2)
  - Dropout (0.25)

- **Fully Connected Layers**:
  - Flatten layer
  - Dense layer (512 neurons, ReLU activation)
  - Dropout (0.5)
  - Output layer (2 neurons, Softmax activation)

**Total Parameters**: Progressive feature extraction from 32 to 256 filters across layers

## 🔧 Technical Features

- **Image Preprocessing**: Grayscale conversion and resizing to 50×50 pixels
- **Data Augmentation**: Shuffled training data for better generalization
- **Regularization Techniques**:
  - Dropout layers to prevent overfitting
  - Batch normalization for stable training
- **Early Stopping**: Custom callback stops training at 95% validation accuracy
- **Optimization**: Adam optimizer with categorical cross-entropy loss
- **Batch Processing**: 32-image batches with TensorFlow Dataset API

## 📊 Training Configuration

- **Image Size**: 50×50 pixels (grayscale)
- **Batch Size**: 32
- **Max Epochs**: 50
- **Early Stopping Threshold**: 95% validation accuracy
- **Training/Validation Split**: 12,500 images each

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow opencv-python numpy pandas tqdm
```

### Dataset Structure

Organize your dataset as follows:
```
train/
  ├── cat.0.jpg
  ├── cat.1.jpg
  ├── dog.0.jpg
  ├── dog.1.jpg
  └── ...
test1/
  ├── 1.jpg
  ├── 2.jpg
  └── ...
```

### Configuration

Update the directory paths in `code.py`:

```python
TRAIN_DIR = 'path/to/your/train/folder'
TEST_DIR = 'path/to/your/test/folder'
```

### Training the Model

Run the training script:

```bash
python code.py
```

The model will:
1. Load and preprocess images
2. Train on the dataset with progress bars (tqdm)
3. Stop early if 95% validation accuracy is reached
4. Save the trained model as `dog_cat_classifier.h5`

## 📁 Project Files

- `code.py` - Main training script with model architecture and data processing
- `dog_cat_classifier.h5` - Saved trained model (generated after training)

## 🧠 Key Functions

- `label_img(img)` - Extracts labels from filenames using one-hot encoding
- `create_train_data()` - Loads, processes, and labels training images
- `process_test_data()` - Processes test images without labels
- `EarlyStoppingAtValAcc` - Custom callback for early stopping at target accuracy

## 📈 Model Performance

- Target validation accuracy: **95%**
- Early stopping prevents overfitting
- Model summary available after training

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework
- **OpenCV (cv2)** - Image processing
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **tqdm** - Progress visualization

## 📝 Notes

- Images are converted to grayscale to reduce computational complexity
- The model uses categorical cross-entropy for binary classification
- Dataset paths are configured for local Windows environment (update as needed)
- Inspired by GeeksForGeeks image processing tutorials

## 🔮 Future Improvements

- Add data augmentation (rotation, flipping, zoom)
- Experiment with larger image sizes
- Implement transfer learning with pre-trained models (VGG16, ResNet)
- Add prediction visualization
- Deploy model as a web application

## 📄 License

This project is open source and available for educational purposes.

## 👨‍💻 Author

**millingtonsully**

---

*Built as part of an Honors Programming project*