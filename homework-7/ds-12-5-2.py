import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 读取图像
image_path = ".\\test.jpg"
original_image = Image.open(image_path)

# 定义目标大小
target_size = (224, 224)

# 图像转换
transform = transforms.Compose([
    transforms.Resize(target_size),
    transforms.ToTensor()
])

# 应用转换
resized_image = transform(original_image)

# 显示原始图像和调整大小后的图像
plt.subplot(1, 2, 1)
plt.imshow(transforms.ToTensor()(original_image).permute(1, 2, 0))
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(resized_image.permute(1, 2, 0))
plt.title('Resized Image')

plt.show()
