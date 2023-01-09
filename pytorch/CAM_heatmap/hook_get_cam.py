import numpy as np
import cv2
import torch 
from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F 

def hook_feature(module, input, output): # hook注册, 响应图提取
    print("hook input",input[0].shape)
    features_blobs.append(output.data.cpu().numpy())
 
def returnCAM(feature_conv, weight_softmax, class_idx, size_upsample):
    # 生成CAM图: 输入是feature_conv和weight_softmax 
    bz, nc, h, w = feature_conv.shape  
    output_cam = []
    for idx in class_idx:
        # feature_conv和weight_softmax 点乘(.dot)得到cam
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w))) 
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


if __name__ == '__main__':

    size_upsample = (224, 224)
    #1. imput image process
    normalize = transforms.Normalize(
     mean=[0.485, 0.456, 0.406],
     std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
     transforms.Resize(size_upsample),
     transforms.ToTensor(),
     normalize])
    img_name = './cam_img/IMG_7052.JPG'
    img_pil = Image.open(img_name)
    img = cv2.imread(img_name)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))

    #2. 导入res18 pretrain, 也可自行定义net结构然后导入.pth
    # net = models.resnet18(pretrained=True) 
    net = models.resnet18(pretrained=False)
    net.load_state_dict(torch.load('./resnet18-f37072fd.pth'), strict=True)
    net.eval()
    # print(net)

    #3. 获取特定层的feature map
    #3.1. hook the feature extractor
    features_blobs = []
    finalconv_name = 'layer4'
    # 对layer4层注册, 把layer4层的输出加入features
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    print(net._modules)

    #3.2. 得到weight_softmax
    params = list(net.parameters()) # 将参数变换为列表 按照weights bias 排列 池化无参数
    weight_softmax = np.squeeze(params[-2].data.numpy()) # 提取softmax 层的参数 （weights，-1是bias）

    #4. imput img inference
    logit = net(img_variable)
    h_x = F.softmax(logit, dim=1).data.squeeze()  
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # features_blobs[0], weight_softmax点乘得到CAM
    CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[2]], size_upsample)

    # 将图片和CAM拼接在一起展示定位结果结果
    img = cv2.resize(img, size_upsample)
    height, width, _ = img.shape
    # 生成热度图
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    cv2.imwrite('./heatmap.jpg', heatmap)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite('./CAM.jpg', result)