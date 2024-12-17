import urllib.request
import cv2

from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, TensorDataset
import os
import seaborn as sns
from statistics import mode


class CNN2(nn.Module):
    def __init__(self, num_classes):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.dropout3 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(128 * 6 * 6, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv5(x)))
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = self.dropout3(x)
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)

model = CNN2(7)
model.load_state_dict(torch.load('./model/fer_model_52.pth'))
model.to(device)

X = torch.rand(size=(1,48,48), dtype=torch.float32)
for layer in iter(model):
    X = layer(X)
    print(layer.__class__.__name__,'out_put shape: \t', X.shape)

def process_live_video(face_Cascade):

    print("Model has been loaded")

    frame_window = 10

    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    emotion_window = []

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cap.set(cv2.CAP_PROP_AUDIO_POS, 0.3)
        # 将frame转换成灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # 裁剪区域

            face_img = frame[y:y + h, x:x + w]

            # 将face_img转换成PIL
            face_img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_img_resized = face_img_pil.resize((48, 48))
            face_img_gray = face_img_resized.convert('L')

            # 将PIL转换成numpy中的array
            face_img_np = np.array(face_img_gray)
            face_img_tensor = torch.tensor(face_img_np, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(
                'cuda' if torch.cuda.is_available() else 'cpu')

            # 将face image送入模型
            with torch.no_grad():
                outputs = model(face_img_tensor)

            # 得到预测的表情标签
            _, predicted = torch.max(outputs, 1)
            predicted_label = class_names[predicted.item()]

            # 在脸上化矩形框，显示预测的标签
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            emotion_window.append(predicted_label)

            if len(emotion_window) >= frame_window:
                emotion_window.pop(0)

            try:
                # 获得出现次数最多的分类
                emotion_mode = mode(emotion_window)
            except:
                continue

            # 用橙色显示主要文字
            cv2.putText(frame, emotion_mode, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 2)

            # 创造橙色效果
            cv2.putText(frame, emotion_mode, (x + 1, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 2)
            cv2.putText(frame, emotion_mode, (x + 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 2)

        print(frame.shape)

        cv2.imshow('FRAME', frame)

        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()


test_folder = 'C:\\Users\\XIEMIAN\\Downloads\\Compressed\\archive\\test'

# 从文件夹读取image和label
def load_images_from_folder(folder):
    images = []
    labels = []
    for emotion_folder in os.listdir(folder):
        label = emotion_folder
        for filename in os.listdir(os.path.join(folder, emotion_folder)):
            img = cv2.imread(os.path.join(folder, emotion_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(label)
    return images, labels

test_images, test_labels = load_images_from_folder(test_folder)

# 将list转换成numpy中的array
test_images = np.array(test_images)
test_labels = np.array(test_labels)

# 检查数据集的shape
print("Test images shape:", test_images.shape)
print("Test labels shape:", test_labels.shape)

#testing data
X_test_tensor = torch.tensor(test_images)
X_test_tensor = X_test_tensor.float()
X_test_tensor = torch.unsqueeze(X_test_tensor, 1)

print('X_test_tensor.shape: ',X_test_tensor.shape, 'X_test_tensor.dtype: ', X_test_tensor.dtype)

#testing data
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(test_labels)
y_test_tensor = torch.tensor(y_test_encoded)

class_names = label_encoder.classes_
print(class_names)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

true_labels, predicted_labels = [], []

def test():

    model.eval()

    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            # 前馈

            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            # 得到预测值
            _, predicted = torch.max(outputs.data, 1)

            # 将实际标签和真实标签添加到list
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # 算samples的总数
            total += labels.size(0)

            # 算正确的预测数
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        print('Accuray on test data is {:.2f}%'.format(accuracy))

test()

# 得到原始的类名
class_names = label_encoder.classes_

# 创造混淆矩阵
conf_matrix = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(12, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False,
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
process_live_video(face_cascade)