import random
import os
import warnings

import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
from sklearn import metrics
import torch.nn.functional as F
import math

def read_train_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    category = [cls for cls in os.listdir(root) if os.path.isdir(os.path.join(root, cls))]
    category.sort()
    class_indices = dict((k, v) for v, k in enumerate(category))

    train_images_path = []
    train_images_label = []

    for cls in category:
        cls_path = os.path.join(root, cls)
        cls_path_MRI = os.path.join(cls_path, "MRI")
        cls_path_PET = os.path.join(cls_path, "PET")
        images_MRI = [os.path.join(cls_path_MRI, i) for i in os.listdir(cls_path_MRI)]
        images_PET = [os.path.join(cls_path_PET, i) for i in os.listdir(cls_path_PET)]

        image_class = class_indices[cls]

        for i in range(0, len(images_MRI)):
            list_of = []
            if images_MRI[i].split("\\")[-1] == images_PET[i].split("\\")[-1]:
                list_of.append(images_MRI[i])
                list_of.append(images_PET[i])

            train_images_path.append(list_of)
            train_images_label.append(image_class)

    print("{} images for test.".format(len(train_images_path)))
    # print("361 images for test.")
    return train_images_path, train_images_label


class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        MRI = nib.load(self.images_path[index][0])
        PET = nib.load(self.images_path[index][1])

        imgdata_MRI = MRI.get_fdata().astype(np.float32)
        imgdata_PET = PET.get_fdata().astype(np.float32)

        # 归一化
        imgdata_MRI = (imgdata_MRI - imgdata_MRI.min()) / (imgdata_MRI.max() - imgdata_MRI.min() + 1e-8)
        imgdata_PET = (imgdata_PET - imgdata_PET.min()) / (imgdata_PET.max() - imgdata_PET.min() + 1e-8)

        data_MRI = torch.from_numpy(imgdata_MRI).float()
        data_PET = torch.from_numpy(imgdata_PET).float()

        label = self.images_class[index]
        return data_MRI, data_PET, label



def val_Model(model,testloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    valloss = 0  # 训练集损失
    accuracy = 0  # 准确度
    sum1 = 0

    with torch.no_grad():  # 关闭自动求导的功能来节约显存与内存
        truelist = []  # 真实列表
        scorelist = []  # 得分列表
        model.eval()
        for id, (MRI, PET, target) in enumerate(testloader):  # testloader与上次一致注意测试集只需要一个epoch将所有数据运行一遍即可
            # 与train一致也是按照patch进行测试
            torch.cuda.empty_cache()  # 运行前清空缓存
            MRI = MRI.to(device)
            PET = PET.to(device)
            image = torch.stack([MRI, PET], dim=1)
            target = target.cuda()  # 将图像数据和CLASS类别加载到GPU上等待运行
            output = model(image)  # 数据加载到网络得到输出m
            if id==1:
                print(output)

            valloss = valloss + F.cross_entropy(output, target, weight=torch.tensor([3.5, 6.5]).cuda()) * len(
                target)  # 与上面train一致
            if math.isnan(valloss):
                print("valloss is NaN")
                quit()
            m = output.max(1)[1]  # max(1)表示按行比较每行的大小并提出来作为一行,从而形成一个数组m为预测结果
            sum1 += m.sum().item()  # 由于使用0与1来表示预测类型所以直接将所有预测结果数据加起来即可得到CN类型数据的个数
            accuracy += m.eq(target).sum().item()  # m中预测正确的数据个数

            truelist = truelist + target.tolist()  # 将target（真实值）改为列表加入到truelist中去
            scorelist = scorelist + [row[1] for row in output.tolist()]  # 将测试机最终结果加入到scorelist中去(实际上为0-1的一群数)
        accuracy = accuracy / len(testloader.dataset)  # 求出总的准确率
        print("Accuracy:", accuracy, "Val Loss:", valloss.item() / len(testloader.dataset), "Purity:",
              sum1 / len(testloader.dataset))  # 分别表示测试集的准确度 损失和纯度
        return accuracy


def test_Model(model,testloader,epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    validloss=0
    valloss=0#训练集损失
    accuracy=0#准确度
    sum1=0
    sen_n=0
    sen_d=0
    spe_n=0
    spe_d=0
    model.eval()
    with torch.no_grad():#关闭自动求导的功能来节约显存与内存

        sen=0
        spe=0
        truelist=[]#真实列表
        scorelist=[]#得分列表
        for id,(MRI,PET,target) in enumerate(testloader):#testloader与上次一致注意测试集只需要一个epoch将所有数据运行一遍即可
            #与train一致也是按照patch进行测试
            torch.cuda.empty_cache()#运行前清空缓存
            MRI = MRI.to(device)
            PET = PET.to(device)
            image = torch.stack([MRI, PET], dim=1)
            target=target.cuda()#将图像数据和CLASS类别加载到GPU上等待运行
            output=model(image)#数据加载到网络得到输出m
            valloss=valloss+F.cross_entropy(output,target,weight=torch.tensor([3.5,6.5]).cuda())*len(target)#与上面train一致


            m=output.max(1)[1]#max(1)表示按行比较每行的大小并提出来作为一行,从而形成一个数组m为预测结果
            # print(m)
            # print(output)
            sum1+=m.sum().item()#由于使用0与1来表示预测类型所以直接将所有预测结果数据加起来即可得到CN类型数据的个数
            accuracy+=m.eq(target).sum().item()#m中预测正确的数据个数
            truelist=truelist+target.tolist()#将target（真实值）改为列表加入到truelist中去
            scorelist=scorelist+[row[1] for row in output.tolist()]#将测试机最终结果加入到scorelist中去(实际上为0-1的一群数)

        accuracy=accuracy/len(testloader.dataset)#求出总的准确率
        print("Accuracy:",accuracy,"Test Loss:",valloss.item()/len(testloader.dataset),"Purity:",sum1/len(testloader.dataset))#分别表示测试集的准确度 损失和纯度
        tn, fp, fn, tp=metrics.confusion_matrix(truelist, np.array(scorelist)>0.5).ravel()
        auc=metrics.roc_auc_score(truelist,scorelist)#计算准确率
        sen=tp / (tp + fn)#二（测试结果）中cn预测的占比
        spe=tn / (tn + fp)#二（测试结果）中ad预测的占比
    return (valloss,accuracy,auc,sen,spe)



