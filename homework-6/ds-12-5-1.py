import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

class ColorToGrayCNN(nn.Module):
    def __init__(self):
        super(ColorToGrayCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return torch.mean(self.conv1(x), dim=1, keepdim=True)

# 读取彩色图像
image_path = ".\\test.jpg"
color_image = Image.open(image_path)

original_size = color_image.size

# 定义图像转换
transform = transforms.Compose([
    transforms.Resize((original_size[1], original_size[0])),  # 调整图像大小
    transforms.ToTensor()           # 转换为张量
])

# 将图像转换为张量并添加批处理维度
input_tensor = transform(color_image).unsqueeze(0)

# 创建模型实例
model = ColorToGrayCNN()

# 将彩色图像转换为灰度图像
gray_image = model(input_tensor)

# 显示原始图像和灰度图像
plt.subplot(1, 2, 1)
plt.imshow(transforms.ToTensor()(color_image).permute(1, 2, 0))
plt.title('Color Image')

plt.subplot(1, 2, 2)
plt.imshow(gray_image.squeeze().detach().numpy(), cmap='gray_r')
plt.title('Grayscale Image')

plt.show()
