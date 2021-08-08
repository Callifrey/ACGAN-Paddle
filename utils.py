import paddle
import numpy as np
from PIL import Image
import os

# 计算当前分类准确率
def compute_acc(preds, labels):
    correct = 0
    preds_ = paddle.argmax(preds, axis=1)
    correct = paddle.to_tensor(np.sum(np.array(preds_ == labels)))
    acc = float(correct) / float(labels.shape[0]) * 100.0
    return acc

def save_samples(images, path):
    images = paddle.nn.functional.pad(images, pad=[2, 2, 2, 2])
    b, c, h, w = images.shape
    results = np.zeros((1, 3, 10*h, 10*w))
    count = 0
    for i in range(10):
        for j in range(10):
            results[:, :, i*h:(i+1)*h, j*w:(j+1)*w] = images[count].unsqueeze(0)
            count += 1
    results = 255 * (results + 1) / 2
    result = np.array(results[0].transpose(1, 2, 0), dtype=np.uint8)
    save_result = Image.fromarray(result)
    save_result.save(path)


def get_same_class_samples(net, opt, cal=0):
    net.eval()
    # 随机获取噪声
    noise_ = paddle.randn(shape=[100, opt.nz - opt.num_classes])
    label = paddle.zeros(shape=[opt.num_classes])
    label[cal] = 1
    noise = paddle.zeros(shape=[100, opt.nz])
    noise[:, 0:opt.nz - opt.num_classes] = noise_
    noise[tuple(np.arange(100)), opt.nz - opt.num_classes:] = label
    img = net(noise)
    result = np.zeros((1, 3, opt.imageSize * opt.num_classes, opt.imageSize * 10))
    for i in range(10):
        for j in range(10):
            result[:, :, j * opt.imageSize:(j + 1) * opt.imageSize, i * opt.imageSize:(i + 1) * opt.imageSize] = img[i*10+j].squeeze(0)
    result = 255 * (result + 1) / 2
    result = np.array(result[0].transpose(1, 2, 0), dtype=np.uint8)
    save_result = Image.fromarray(result)
    save_result.save(os.path.join(opt.class_result_path, "class_samples_class_{}.png".format(cal)))
