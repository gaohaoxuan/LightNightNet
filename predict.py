from DenseNet import DenseNet
import torch.nn as nn
import torch, os, cv2, json
import numpy as np
from PIL import Image
from torchvision import transforms

model = DenseNet(nClasses=100)
model = model.cpu()
model = nn.DataParallel(model)
model = torch.load('/export/zhangjiangfeng/gaohaoxuan/LightWeightNet/DenselyNet/checkpoint/model.pth', map_location='cpu')
model = model.module.to(torch.device('cpu'))

root_dir = '/export/zhangjiangfeng/gaohaoxuan/LightWeightNet/data/val'
imgs = os.listdir(root_dir)
imgs_path = []
for img in imgs:
    imgs_path.append(os.path.join(*[root_dir, img]))

# 加载json文件
garbage_dict = open('/export/zhangjiangfeng/gaohaoxuan/LightWeightNet/data/cifar10_dict.json', 'r')
label_dict = garbage_dict.read()
label_dict = json.loads(label_dict)
# print(label_dict)
garbage_dict.close()
# for i in range(0, 10):
#     print(label_dict.get('%s'%(i)))
# list_values = list(label_dict.values())
# print(type(list_values.index('shoes')))

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
model.eval()
with torch.no_grad():
    for img_path in imgs_path:
        label = img_path.split('/')[-1][:-5]
        img = Image.open(img_path)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        pred = model(img)
        print(pred.shape)
        _, predicted = torch.max(pred.data, dim=1)
        predicted = predicted.numpy()
        # print(predicted)
        # print(type(int(predicted[0])))
        print("真实标签：%s, 预测为：%s"%(label, label_dict.get('%s'%(int(predicted)))))
