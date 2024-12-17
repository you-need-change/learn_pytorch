import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler  # pip install scikit-learn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class OttoDataset(Dataset):
    def __init__(self, feature_filepath, label_filepath=None, mode='train', scaler=None):
        super(OttoDataset, self).__init__()

        # Load the dataset into a pandas dataframe.
        data = pd.read_csv(feature_filepath)

        if mode == 'train':
            # Extract the numeric part of the class labels, convert to integers, and shift to zero-based indexing.
            self.labels = torch.tensor(data.iloc[:, -1].apply(lambda x: int(x.split('_')[-1]) - 1).values,
                                       dtype=torch.long)

            # Initialize the StandardScaler.
            # StandardScaler will normalize the features (i.e., each column of the dataset)
            # by subtracting the mean and dividing by the standard deviation.
            # This centers the feature columns at mean 0 with standard deviation 1.
            self.scaler = StandardScaler()

            # Select all columns except 'id' and 'target' for features.
            # Then apply the scaler to standardize them.
            features = data.iloc[:, 1:-1].values
            self.features = torch.tensor(self.scaler.fit_transform(features), dtype=torch.float32)

        elif mode == 'test':
            features = data.iloc[:, 1:].values

            # Apply the same scaling as on the training set to the test set features. use self.scaler.transform
            self.scaler = scaler if scaler is not None else StandardScaler()
            self.features = torch.tensor(self.scaler.transform(features), dtype=torch.float32)

            if label_filepath is not None:
                label_data = pd.read_csv(label_filepath)
                # Assuming the first column after 'id' are one-hot encoded class labels,
                # find the index of the max value in each row which corresponds to the predicted class.
                self.labels = torch.tensor(label_data.iloc[:, 1:].values.argmax(axis=1), dtype=torch.long)

            else:
                self.labels = None

        # If neither 'train' nor 'test' mode is specified, raise an error.
        else:
            raise ValueError("Mode must be 'train' or 'test'")

        # Store the length of the dataset.
        self.len = len(self.features)

    def __len__(self):
        # When len(dataset) is called, return the length of the dataset.
        return self.len

    def __getitem__(self, index):
        # This method retrieves the features and label of a specified index.
        return self.features[index], self.labels[index] if self.labels is not None else -1


class FullyConnectedModel(torch.nn.Module):
    def __init__(self, input_features, output_classes):
        super(FullyConnectedModel, self).__init__()

        # 定义网络层
        self.fc1 = torch.nn.Linear(input_features, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 32)
        self.fc4 = torch.nn.Linear(32, output_classes)

        # 可以选择增加更多的层

        # 定义 dropout 层，可以减少过拟合
        self.dropout = torch.nn.Dropout(p=0.3)

        # 定义 batchnorm 层，帮助稳定学习过程
        self.batchnorm1 = torch.nn.BatchNorm1d(128)
        self.batchnorm2 = torch.nn.BatchNorm1d(64)
        self.batchnorm3 = torch.nn.BatchNorm1d(32)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.batchnorm3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x


def train(epoch, train_loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, targets = data

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 300 == 299:
            print('Epoch:[{}/{}], Loss:{:.4f}'.format(epoch + 1, batch_idx + 1, running_loss / 300))

    # 计算平均损失
    average_loss = running_loss / len(train_loader)
    return average_loss


def test(test_loader, model):
    model.eval()
    correct = 0.0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            targets = targets.to(device)
            _, predicted = torch.max(outputs.data, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = 100 * (correct / total)
    print("Accuracy on test data is {:.2f}".format(accuracy))
    return accuracy


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare dataset
    train_dataset = OttoDataset(feature_filepath='./data/Otto/train.csv', mode='train')
    scaler = train_dataset.scaler
    test_dataset = OttoDataset(feature_filepath='./data/Otto/test.csv', label_filepath='./data/Otto/otto_correct_submission.csv', mode='test', scaler=scaler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Design model
    model = FullyConnectedModel(input_features=93, output_classes=9).to(device)

    # Construct loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)

    # Train and Test
    train_losses = []
    test_accuracies = []

    num_epochs = 100
    for epoch in range(num_epochs):
        train_loss = train(epoch, train_loader, model, criterion, optimizer)
        train_losses.append(train_loss)

        if epoch % 2 == 0 or epoch == num_epochs - 1:
            test_accuracy = test(test_loader, model)
            test_accuracies.append(test_accuracy)

        # Update the learning rate
        scheduler.step()

    # Save model parameters for future use
    torch.save(model.state_dict(), 'model/09_kaggle_OttoDataset_model.pth')

    # Visualize
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(range(0, 101, 2), test_accuracies, label='Test Accuracy')  # Adjust x-axis for test accuracy
    plt.title('Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.show()