from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import torch 


def hook_feature(module, input, output): # hook注册, 实现响应图提取
    print("hook input",input[0].shape)
    features_blobs.append(output.data.cpu().numpy())


# 生成CAM图的函数，完成权重和feature相乘操作，最后resize成上采样
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (224, 224)
    bz, nc, h, w = feature_conv.shape # 获取feature_conv特征的尺寸
    output_cam = []
    #lass_idx为预测分值较大的类别的数字表示的数组，一张图片中有N类物体则数组中N个元素
    for idx in class_idx:
    # weight_softmax中预测为第idx类的参数w乘以feature_map(为了相乘，故reshape了map的形状)

        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w))) # 把原来的相乘再相加转化为矩阵
                                                                    # w1*c1 + w2*c2+ .. -> (w1,w2..) * (c1,c2..)^T -> (w1,w2...)*((c11,c12,c13..),(c21,c22,c23..))
        # 将feature_map的形状reshape回去
        cam = cam.reshape(h, w)
        # 归一化操作（最小的值为0，最大的为1）
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        # 转换为图片的255的数据
        cam_img = np.uint8(255 * cam_img)
        # resize 图片尺寸与输入图片一致
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam



model_id = 2
# 选择使用的网络
if model_id == 1:
 net = models.squeezenet1_1(pretrained=True)
 finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
 net = models.resnet18(pretrained=True) 
 finalconv_name = 'layer4'
elif model_id == 3:
 net = models.densenet161(pretrained=True)
 finalconv_name = 'features'
net.eval()
# print(net)

# 获取特定层的feature map
# hook the feature extractor
features_blobs = []
# 对layer4层注册，把layer4层的输出append到features里面
net._modules.get(finalconv_name).register_forward_hook(hook_feature) # 注册到finalconv_name,如果执行net()的时候，                                                        # 会对注册的钩子也执行，这里就是执行了 layer4()
print(net._modules)

# 得到softmax weight,
params = list(net.parameters()) # 将参数变换为列表 按照weights bias 排列 池化无参数
weight_softmax = np.squeeze(params[-2].data.numpy()) # 提取softmax 层的参数 （weights，-1是bias）


# 数据处理，先缩放尺寸到（224*224），再变换数据类型为tensor,最后normalize
normalize = transforms.Normalize(
 mean=[0.485, 0.456, 0.406],
 std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
 transforms.Resize((224,224)),
 transforms.ToTensor(),
 normalize
])

img_pil = Image.open('./cam_img/IMG_7052.JPG')

img_tensor = preprocess(img_pil)
# 处理图片为Variable数据
img_variable = Variable(img_tensor.unsqueeze(0))
# 将图片输入网络得到预测类别分值
logit = net(img_variable)

# 使用softmax打分
h_x = F.softmax(logit, dim=1).data.squeeze() # 分类分值
# 对分类的预测类别分值排序，输出预测值和在列表中的位置
probs, idx = h_x.sort(0, True)
# 转换数据类型
probs = probs.numpy()
idx = idx.numpy()

# 输出与图片尺寸一致的CAM图片
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[2]])

# 将图片和CAM拼接在一起展示定位结果结果
img = cv2.imread('./cam_img/IMG_7052.JPG')
img = cv2.resize(img,(224,224))
height, width, _ = img.shape
# 生成热度图
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
cv2.imwrite('./heatmap.jpg', heatmap)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('./CAM.jpg', result)