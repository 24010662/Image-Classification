# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

Image classification is a fundamental problem in computer vision, where the goal is to assign an input image to one of the predefined categories. Traditional machine learning models rely heavily on handcrafted features, whereas Convolutional Neural Networks (CNNs) automatically learn spatial features directly from pixel data.

In this experiment, the task is to build a Convolutional Deep Neural Network (CNN) to classify images from the FashionMNIST dataset into their respective categories. The trained model will then be tested on new/unseen images to verify its effectiveness.

Dataset
The FashionMNIST dataset consists of 70,000 grayscale images of size 28 × 28 pixels.

The dataset has 10 classes, representing different clothing categories such as T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot.

Training set: 60,000 images.

Test set: 10,000 images.

Images are preprocessed (normalized and converted to tensors) before being passed into the CNN.

## Neural Network Model

<img width="1121" height="543" alt="image" src="https://github.com/user-attachments/assets/5539d1bb-afd4-4a45-a711-0fdbdbd44413" />


## DESIGN STEPS

### STEP 1:
Import the required libraries such as PyTorch, Torchvision, NumPy, and Matplotlib.

### STEP 2:
Load the FashionMNIST dataset and apply transformations (normalization, tensor conversion).

### STEP 3:
Split the dataset into training and testing sets.

### STEP 4:
Define the CNN architecture with convolutional, pooling, and fully connected layers.

### STEP 5:
Specify the loss function (CrossEntropyLoss) and optimizer (Adam).

### STEP 6:
Train the model using forward pass, loss computation, backpropagation, and parameter updates.

### STEP 7:
Evaluate the model on the test dataset and calculate accuracy.

### STEP 8:
Test the trained model on new/unseen FashionMNIST images.


## PROGRAM

### Name:VUTUKURI SAI KUMAR REDDY
### Register Number:212224230307
```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)  # Changed in_channel to in_channels and out_channel to out_channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Changed in_channel to in_channels and out_channel to out_channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  # Changed in_channel to in_channels and out_channel to out_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # write your code here
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



```

```
# Initialize model, loss function, and optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

```
# Train the Model
def train_model(model, train_loader, num_epochs=3):

    # write your code here
    print('Name: VUTUKURI SAI KUMAR REDDY')
    print('Register Number: 212224230307')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')




```

## OUTPUT
### Training Loss per Epoch

<img width="294" height="119" alt="image" src="https://github.com/user-attachments/assets/90125218-a876-4d6c-a451-85d1967e9dcd" />


### Confusion Matrix

<img width="867" height="793" alt="image" src="https://github.com/user-attachments/assets/f0a0230b-7800-47d4-88e5-4d26812cadb5" />


### Classification Report

<img width="552" height="417" alt="image" src="https://github.com/user-attachments/assets/ebce74d5-8f72-4bd9-86a8-501e9aeb9f83" />



### New Sample Data Prediction

<img width="490" height="564" alt="image" src="https://github.com/user-attachments/assets/887de437-411a-4118-956d-464d8b19446d" />


## RESULT
The Convolutional Neural Network was successfully implemented for FashionMNIST image classification. The model achieved good accuracy on the test dataset and produced reliable predictions for new images, proving its effectiveness in extracting spatial features from images.
