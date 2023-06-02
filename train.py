import torch, torchvision, time, os, gc
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from DenseNet import DenseNet
from tqdm import tqdm

transform_train = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_dataset = torchvision.datasets.CIFAR10(root='/export/zhangjiangfeng/gaohaoxuan/LightWeightNet/DenselyNet', train=True, transform=transform_train, download=False)

transform_test = transforms.Compose([transforms.Resize(256),         #transforms.Scale(256)
                                     transforms.CenterCrop(224), 
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
test_dataset = torchvision.datasets.CIFAR10(root='/export/zhangjiangfeng/gaohaoxuan/LightWeightNet/DenselyNet', train=False, transform=transform_test, download=False)
# print(len(train_dataset))
# print(len(test_dataset))

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# print(len(test_dataloader))

load_epoch = 36
max_epoch = 100
num_train = len(train_dataloader)
# print(num_train)
num_val = len(test_dataloader)
# print(num_val)

device_ids = [0, 1, 2, 3]
model = DenseNet()
model = nn.DataParallel(model, device_ids=device_ids).to(device_ids[0])

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

save_pth = "./checkpoint/"
if not os.path.exists(save_pth):
    os.makedirs(save_pth)
    print(f"{save_pth} 文件夹已创建")
else:
    print(f"{save_pth} 文件夹已存在")

if load_epoch > 1:
    path = save_pth + str(load_epoch) + ".pth"
    model = torch.load(path)

for epoch in range(load_epoch, max_epoch+1):
    # 强制垃圾回收
    gc.collect()
    torch.cuda.empty_cache()
    # 此处不一定能快 需要保证数据一致 该数据集可以使用
    torch.backends.cudnn.benchmark = True

    if epoch != load_epoch:
        path = save_pth + str(epoch) + ".pth"
        torch.save(model, path)
    
    avg_loss = 0.0
    acc_batch = 0.0
    model.train()
    for _, samples in enumerate(tqdm(train_dataloader, desc='train')):
        train_input, train_label = samples
        # print(train_input.shape, train_label.shape)
        train_input = train_input.to('cuda:0', non_blocking=True)
        train_label = train_label.to('cuda:0', non_blocking=True)

        pred = model(train_input)
        # print(pred)
        # print(train_label.shape)
        optimizer.zero_grad()
        loss = loss_fn(pred, train_label)
        loss.backward()
        optimizer.step()

        avg_loss += loss.detach().data
        # 记录准确率
        num_batch = train_label.size()[0]#图片个数
        _, predicted = torch.max(pred.data, dim=1)
        correct_batch = (predicted == train_label).sum().item()#预测正确的数目
        acc_batch = 100 * correct_batch / num_batch
    print("Epoch: ", epoch, "Avg_Loss: ", avg_loss / num_train)
    print("Epoch: ", epoch, "acc_rate: ", acc_batch)

    model.eval()
    with torch.no_grad():
        val_time = 0.0
        for _, samples in enumerate(tqdm(test_dataloader)):
            val_input, val_label = samples

            strat = time.time()
            pred = model(val_input)
            end = time.time()
            val_time += end - strat

            # 记录准确率
            num_batch = val_label.size()[0]#图片个数
            _, predicted = torch.max(pred.data.cpu(), dim=1)
            correct_batch = (predicted == val_label).sum().item()#预测正确的数目
            acc_batch = 100 * correct_batch / num_batch
        with open('./data.txt', 'a') as f:
            f.write(str(acc_batch) + '\t' + str(val_time / num_val) + '\n')
torch.save(model, save_pth + "model.pth")