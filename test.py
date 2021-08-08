import argparse
from utils import *
from network import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/media/gallifrey/DJW/Dataset/Imagenet/train', help='path to dataset')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--check_path', default='./checkpoints', help='folder to output images and model checkpoints')
parser.add_argument('--result_path', type=str, default='./results', help='folder to save results picture')
parser.add_argument('--class_result_path', type=str, default='./class_results', help='folder to save results picture')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')
parser.add_argument('--which_epoch', type=int, default=495, help='Test checkpoints')

opt = parser.parse_args()
print(opt)

# 加载预训练模型
G = Generator(nz=opt.nz)
state_dict = paddle.load(os.path.join(opt.check_path, 'G_{}.pdparams'.format(opt.which_epoch)))
G.load_dict(state_dict)
G.eval()

# 生成随机噪声
batch_size = opt.batchSize
noise = paddle.randn(shape=[batch_size, opt.nz])
label = paddle.randint(0, opt.num_classes, shape=[batch_size])
class_onehot = paddle.zeros([batch_size, opt.num_classes])
class_onehot[np.arange(batch_size), label] = 1
noise[0:batch_size, :opt.num_classes] = class_onehot[0:batch_size]
noise = paddle.reshape(noise, shape=[batch_size, opt.nz, 1, 1])

# 生成fake
fake = G(noise)
save_samples(fake, path=os.path.join(opt.result_path, 'sample_epoch_{}.png'.format(opt.which_epoch)))