import torch
import torch.nn as nn
import torch.nn.functional as F

class PCN1(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, dilation=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2, stride=1)
        self.rotate = nn.Conv2d(128, 2, kernel_size=1, stride=1)
        self.cls_prob = nn.Conv2d(128, 2, kernel_size=1, stride=1)
        self.bbox = nn.Conv2d(128, 3, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        rotate = F.softmax(self.rotate(x), dim=1)
        bbox = self.bbox(x)
        return cls_prob, rotate, bbox


# caffe output for data shape
# data                        	 (1, 3, 24, 24)
# conv1_1                     	 (1, 16, 11, 11)
# conv2_1                     	 (1, 32, 5, 5)
# conv3_1                     	 (1, 64, 2, 2)
# fc4_1                       	 (1, 128, 1, 1)
# fc4_1_relu4_1_0_split_0     	 (1, 128, 1, 1)
# fc4_1_relu4_1_0_split_1     	 (1, 128, 1, 1)
# fc4_1_relu4_1_0_split_2     	 (1, 128, 1, 1)
# fc5_1                       	 (1, 2, 1, 1)
# cls_prob                    	 (1, 2, 1, 1)
# fc6_1                       	 (1, 2, 1, 1)
# rotate_cls_prob             	 (1, 2, 1, 1)
# bbox_reg_1                  	 (1, 3, 1, 1)

# caffe param
# conv1_1                     	 (16, 3, 3, 3) (16,)
# conv2_1                     	 (32, 16, 3, 3) (32,)
# conv3_1                     	 (64, 32, 3, 3) (64,)
# fc4_1                       	 (128, 64, 2, 2) (128,)
# fc5_1                       	 (2, 128, 1, 1) (2,)
# fc6_1                       	 (2, 128, 1, 1) (2,)
# bbox_reg_1                  	 (3, 128, 1, 1) (3,)

class PCN2(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(40, 70, kernel_size=2, stride=1)
        self.fc = nn.Linear(70*3*3, 140)
        self.rotate = nn.Linear(140, 3)
        self.cls_prob = nn.Linear(140, 2)
        self.bbox = nn.Linear(140, 3)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp(x), inplace=True)
        x = self.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        #x = x.view(batch_size, -1)
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc(x), inplace=True)
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        rotate = F.softmax(self.rotate(x), dim=1)
        bbox = self.bbox(x)
        return cls_prob, rotate, bbox

# caffe output for data shape
# data                        	 (1, 3, 24, 24)
# conv1_2                     	 (1, 20, 22, 22)
# pool1_2                     	 (1, 20, 11, 11)
# conv2_2                     	 (1, 40, 9, 9)
# pool2_2                     	 (1, 40, 4, 4)
# conv3_2                     	 (1, 70, 3, 3)
# fc4_2                       	 (1, 140)
# fc4_2_relu4_2_0_split_0     	 (1, 140)
# fc4_2_relu4_2_0_split_1     	 (1, 140)
# fc4_2_relu4_2_0_split_2     	 (1, 140)
# fc5_2                       	 (1, 2)
# cls_prob                    	 (1, 2)
# fc6_2                       	 (1, 3)
# rotate_cls_prob             	 (1, 3)
# bbox_reg_2                  	 (1, 3)

# caffe param
# conv1_2                     	 (20, 3, 3, 3) (20,)
# conv2_2                     	 (40, 20, 3, 3) (40,)
# conv3_2                     	 (70, 40, 2, 2) (70,)
# fc4_2                       	 (140, 630) (140,)
# fc5_2                       	 (2, 140) (2,)
# fc6_2                       	 (3, 140) (3,)
# bbox_reg_2                  	 (3, 140) (3,)

class PCN3(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(96, 144, kernel_size=2, stride=1)
        self.fc = nn.Linear(144*3*3, 192)
        self.cls_prob = nn.Linear(192, 2)
        self.bbox = nn.Linear(192, 3)
        self.rotate = nn.Linear(192, 1)
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)


    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp1(x), inplace=True)

        x = self.conv2(x)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.mp1(x), inplace=True)

        x = self.conv3(x)
        x = F.relu(self.mp2(x), inplace=True)
        x = F.relu(self.conv4(x), inplace=True)
        #x = x.view(batch_size, -1)
        x = x.reshape(batch_size, -1)
        x = F.relu(self.fc(x), inplace=True)
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        rotate = self.rotate(x)
        bbox = self.bbox(x)
        return cls_prob, rotate, bbox


'''
# valid
class PCN3(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolutional and Pooling Layers
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv2d(96, 144, kernel_size=2, stride=1)
        
        # Fully Connected Layers
        self.fc = nn.Linear(144 * 3 * 3, 192)  # Adjust the input size based on the output size of conv4
        self.cls_prob = nn.Linear(192, 2)
        
        # Regression Outputs
        self.bbox = nn.Linear(192, 3)
        self.rotate = nn.Linear(192, 1)

    def forward(self, x):
        # Conv1 Block
        x = self.conv1(x)  # Output size: (24, 48, 48)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.pool1(x)  # Output size: (24, 24, 24)
        x = F.relu(x)
        
        # Conv2 Block
        x = self.conv2(x)  # Output size: (48, 24, 24)
        x = F.pad(x, (0, 1, 0, 1))
        x = self.pool2(x)  # Output size: (48, 12, 12)
        x = F.relu(x)
        
        # Conv3 Block
        x = self.conv3(x)  # Output size: (96, 12, 12)
        x = self.pool3(x)  # Output size: (96, 6, 6)
        x = F.relu(x)
        
        # Conv4 Block
        x = self.conv4(x)  # Output size: (144, 5, 5) because kernel size is 2
        x = F.relu(x)
        
        # Flatten the tensor for fully connected layers
        x = x.reshape(x.size(0), -1)  # Output size: (batch_size, 144 * 3 * 3)
        
        # FC Layer
        x = self.fc(x)  # Output size: (batch_size, 192)
        x = F.relu(x)
        
        # Classification Output
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        
        # Regression Outputs
        rotate = self.rotate(x)
        bbox = self.bbox(x)
        
        return cls_prob, rotate, bbox
'''
'''
# broken
class PCN3(nn.Module):
    def __init__(self):
        super(PCN3, self).__init__()
        
        # Convolutional and Pooling Layers
        self.conv1 = nn.Conv2d(3, 24, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.conv4 = nn.Conv2d(96, 144, kernel_size=2)
        
        # Fully Connected Layers
        self.fc = nn.Linear(144 * 3 * 3, 192)  # Adjust the input size based on the output size of conv4
        self.cls_prob = nn.Linear(192, 2)
        
        # Regression Outputs
        self.bbox = nn.Linear(192, 3)
        self.rotate = nn.Linear(192, 1)

    def forward(self, x):
        # Conv1 Block
        x = self.conv1(x)  # Output size: (24, 48, 48)
        x = self.pool1(x)  # Output size: (24, 24, 24)
        x = F.relu(x)
        
        # Conv2 Block
        x = self.conv2(x)  # Output size: (48, 24, 24)
        x = self.pool2(x)  # Output size: (48, 12, 12)
        x = F.relu(x)
        
        # Conv3 Block
        x = self.conv3(x)  # Output size: (96, 12, 12)
        x = self.pool3(x)  # Output size: (96, 6, 6)
        x = F.relu(x)
        
        # Conv4 Block
        x = self.conv4(x)  # Output size: (144, 5, 5) because kernel size is 2
        x = F.relu(x)
        
        # Flatten the tensor for fully connected layers
        x = x.reshape(x.size(0), -1)  # Output size: (batch_size, 144 * 3 * 3)
        
        # FC Layer
        x = self.fc(x)  # Output size: (batch_size, 192)
        x = F.relu(x)
        
        # Classification Output
        cls_prob = F.softmax(self.cls_prob(x), dim=1)
        
        # Regression Outputs
        rotate = self.rotate(x)
        bbox = self.bbox(x)
        
        return cls_prob, rotate, bbox
'''


# caffe output for data shape
# data                        	 (1, 3, 48, 48)
# conv1_3                     	 (1, 24, 46, 46)
# pool1_3                     	 (1, 24, 23, 23)
# conv2_3                     	 (1, 48, 21, 21)
# pool2_3                     	 (1, 48, 10, 10)
# conv3_3                     	 (1, 96, 8, 8)
# pool3_3                     	 (1, 96, 4, 4)
# conv4_3                     	 (1, 144, 3, 3)
# fc5_3                       	 (1, 192)
# fc5_3_relu5_3_0_split_0     	 (1, 192)
# fc5_3_relu5_3_0_split_1     	 (1, 192)
# fc5_3_relu5_3_0_split_2     	 (1, 192)
# fc6_3                       	 (1, 2)
# cls_prob                    	 (1, 2)
# bbox_reg_3                  	 (1, 3)
# rotate_reg_3                	 (1, 1)


# caffe param
# conv1_3                     	 (24, 3, 3, 3) (24,)
# conv2_3                     	 (48, 24, 3, 3) (48,)
# conv3_3                     	 (96, 48, 3, 3) (96,)
# conv4_3                     	 (144, 96, 2, 2) (144,)
# fc5_3                       	 (192, 1296) (192,)
# fc6_3                       	 (2, 192) (2,)
# bbox_reg_3                  	 (3, 192) (3,)
# rotate_reg_3                	 (1, 192) (1,)

class PCNTracking(nn.Module):

    def __init__(self):
        super().__init__()
        
        # conv1
        self.conv1_1 = nn.Conv2d(3, 8, kernel_size=3, stride=1)
        self.conv1_2 = nn.Conv2d(8, 8, kernel_size=1, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # conv2
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, stride=1)
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=1, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # conv3
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.conv3_2 = nn.Conv2d(32, 32, kernel_size=1, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # conv4
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=1, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # conv5
        self.conv5_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv5_2 = nn.Conv2d(128, 128, kernel_size=1, stride=1)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # fc layers
        self.fc6 = nn.Linear(128 * 3 * 3, 256)  # assuming input image size is 96x96

        self.fc7 = nn.Linear(256, 2)
        self.bbox_reg = nn.Linear(256, 3)
        self.rotate_reg = nn.Linear(256, 1)
        self.points_reg = nn.Linear(256, 28)

    def forward(self, x):
        batch_size = x.size(0)

        # conv1
        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv1_1(x), inplace=True)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.conv1_2(x), inplace=True)
        x = self.pool1(x)
        
        # conv2
        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv2_1(x), inplace=True)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.conv2_2(x), inplace=True)
        x = self.pool2(x)
        
        # conv3
        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv3_1(x), inplace=True)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.conv3_2(x), inplace=True)
        x = self.pool3(x)
        
        # conv4
        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv4_1(x), inplace=True)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.conv4_2(x), inplace=True)
        x = self.pool4(x)
        
        # conv5
        x = F.pad(x, (1, 1, 1, 1))
        x = F.relu(self.conv5_1(x), inplace=True)
        x = F.pad(x, (0, 1, 0, 1))
        x = F.relu(self.conv5_2(x), inplace=True)
        x = self.pool5(x)
        
        # Flatten the output for the fully connected layers
        #x = x.view(x.size(0), -1)
        x = x.reshape(batch_size, -1)

        # fc layers
        x = F.relu(self.fc6(x), inplace=True)
        
        cls_prob = F.softmax(self.fc7(x), dim=1)
        # BBox regression
        bbox_reg = self.bbox_reg(x)
        # Rotate regression
        rotate_reg = self.rotate_reg(x)
        # Points regression
        points_reg = self.points_reg(x)

        return cls_prob, bbox_reg, rotate_reg, points_reg

import os
from os.path import join as pjoin

def load_model():
    cwd = os.path.dirname(__file__)
    pcn1, pcn2, pcn3, pcn4 = PCN1(), PCN2(), PCN3(), PCNTracking()
    pcn1.load_state_dict(torch.load(pjoin(cwd, 'pth/pcn1_sd.pth')))
    pcn2.load_state_dict(torch.load(pjoin(cwd, 'pth/pcn2_sd.pth')))
    pcn3.load_state_dict(torch.load(pjoin(cwd, 'pth/pcn3_sd.pth')))
    pcn4.load_state_dict(torch.load(pjoin(cwd, 'pth/pcn-tracking.pth')))
    return pcn1, pcn2, pcn3, pcn4