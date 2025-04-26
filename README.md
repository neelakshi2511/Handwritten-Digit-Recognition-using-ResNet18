# Handwritten-Digit-Recognition-using-ResNet18

This implementation focuses on building an image classification model to recognize handwritten pen digits (0–9) using **transfer learning** with a pre-trained **ResNet18** model in **PyTorch**.

The project is implemented in **Google Colab** and involves loading a custom dataset from **Google Drive**, preprocessing images, fine-tuning a pre-trained model, training, evaluating performance, and visualizing the results.


### 1. Mount Google Drive
Mount your Google Drive to access the dataset.

### 2. Set Dataset Paths
Specify the paths to the training and testing datasets stored in your Drive.

### 3. Import Required Libraries
Libraries like PyTorch, torchvision, matplotlib, sklearn, and others are used for model building, evaluation, and visualization.

### 4. Define Image Transformations
Preprocessing steps include:
- Resizing images to 224x224 pixels
- Normalization
- Random horizontal flipping for training augmentation

### 5. Load Dataset
Load the dataset using `torchvision.datasets.ImageFolder` and prepare DataLoaders for both training and testing.

### 6. Model Setup
- Load a **pre-trained ResNet18** model.
- Modify the final fully connected layer to classify 10 classes (digits 0–9).

### 7. Set Loss Function and Optimizer
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: Adam with a learning rate of 0.001

### 8. Train the Model
Train the model for **5 epochs**, recording and plotting training loss over time.

### 9. Evaluate the Model
- Compute **overall test accuracy**.
- **Visualize sample predictions** alongside true labels.
- **Plot confusion matrix** to analyze prediction performance across classes.
- **Generate a detailed classification report** (precision, recall, F1-score for each class).


