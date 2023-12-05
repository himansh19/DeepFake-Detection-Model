import sys
import torch
import cv2
import numpy as np
from torchvision import transforms
from torch import nn
from torchvision import models
# import sys
# import torch
# from torchvision import transforms
# from PIL import Image

# from torch import nn
# from torchvision import models

class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)  # Residual Network CNN
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

def predict(video_data):
    # Decode the video_data bytes using OpenCV
    video_np = np.frombuffer(video_data, dtype=np.uint8)
    cap = cv2.VideoCapture()
    cap.open(video_np)

    # Create a PyTorch transformation for each video frame
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load the model
    model = Model(num_classes=2) 
    model.load_state_dict(torch.load('/content/checkpoint.pt', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()

    # Loop through video frames
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Convert the frame to a PyTorch tensor
        frame_tensor = transform(frame).unsqueeze(0)

        # Perform inference with the model
        with torch.no_grad():
            output = model(frame_tensor)
        predicted_class = torch.argmax(output).item()

        print("Predicted Class:", predicted_class)

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python predict.py <video_path>")
        sys.exit(1)

    # Get the video file path from the command-line arguments
    video_path = sys.argv[1]

    # Call the predict function with the video file path
    predict(video_path)