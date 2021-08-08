import paddle.vision.transforms as tran
from paddle.io import Dataset, DataLoader
import argparse
import paddle.nn as nn
from visualdl import LogWriter

from utils import *
from network import Generator, Discriminator
from dataset import ImageFolder


parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='/media/gallifrey/DJW/Dataset/Imagenet/train', help='path to dataset')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--check_path', default='./checkpoints', help='folder to output images and model checkpoints')
parser.add_argument('--result_path', type=str, default='./results', help='folder to save results picture')
parser.add_argument('--class_result_path', type=str, default='./class_results', help='folder to save fix class results picture')
parser.add_argument('--log_path', type=str, default='./log', help='folder to save log file')
parser.add_argument('--save_freq', type=int, default=5, help='frequency for save')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')


opt = parser.parse_args()
print(opt)

# 加载数据
dataset = ImageFolder(
    root=opt.dataroot,
    transform=tran.Compose([
        tran.Resize(opt.imageSize),
        tran.CenterCrop(opt.imageSize),
        tran.ToTensor(),
        tran.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    classes_idx=(90,100)
)
dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=int(opt.workers), drop_last=True)

# 实例化模型
netG = Generator(opt.nz)
netD = Discriminator(opt.num_classes)

# 定义损失函数
dis_criterion = nn.BCELoss()
aux_criterion = nn.NLLLoss()

real_label = 1
fake_label = 0

# 定义优化器
optimizerG = paddle.optimizer.Adam(learning_rate=opt.lr, beta1=opt.beta1, beta2=0.999, parameters=netG.parameters())
optimizerD = paddle.optimizer.Adam(learning_rate=opt.lr, beta1=opt.beta1, beta2=0.999, parameters=netD.parameters())

avg_loss_D = 0.0
avg_loss_G = 0.0
avg_loss_A = 0.0


with LogWriter(logdir=opt.log_path) as writer:
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) 更新判别器: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            optimizerD.clear_grad()
            real_cpu, label = data
            batch_size = real_cpu.shape[0]
            dis_label = paddle.full([batch_size], real_label)

            dis_output, aux_output = netD(real_cpu)

            dis_errD_real = dis_criterion(dis_output, dis_label)
            aux_errD_real = aux_criterion(aux_output, label)
            errD_real = dis_errD_real + aux_errD_real
            errD_real.backward()
            D_x = dis_output.mean()

            # compute the current classification accuracy
            accuracy = compute_acc(aux_output, label)

            # train with fake
            noise = paddle.randn(shape=[batch_size, opt.nz])
            label = paddle.randint(0, opt.num_classes, shape=[batch_size])
            class_onehot = paddle.zeros([batch_size, opt.num_classes])
            class_onehot[np.arange(batch_size), label] = 1
            noise[0:batch_size, :opt.num_classes] = class_onehot[0:batch_size]
            noise = paddle.reshape(noise, shape=[batch_size, opt.nz, 1, 1])

            fake = netG(noise)
            dis_label = paddle.full([batch_size], fake_label)
            aux_label = label
            dis_output, aux_output = netD(fake.detach())
            dis_errD_fake = dis_criterion(dis_output, dis_label)
            aux_errD_fake = aux_criterion(aux_output, aux_label)
            errD_fake = dis_errD_fake + aux_errD_fake
            errD_fake.backward()
            D_G_z1 = dis_output.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) 更新生成器: maximize log(D(G(z)))
            ###########################
            optimizerG.clear_grad()
            dis_label = paddle.full([batch_size], real_label)
            dis_output, aux_output = netD(fake)
            dis_errG = dis_criterion(dis_output, dis_label)
            aux_errG = aux_criterion(aux_output, aux_label)
            errG = dis_errG + aux_errG
            errG.backward()
            D_G_z2 = dis_output.mean()
            optimizerG.step()

            # 计算平均损失/分类精度
            curr_iter = epoch * len(dataloader) + i
            all_loss_G = avg_loss_G * curr_iter
            all_loss_D = avg_loss_D * curr_iter
            all_loss_A = avg_loss_A * curr_iter
            all_loss_G += errG.item()
            all_loss_D += errD.item()
            all_loss_A += accuracy
            avg_loss_G = all_loss_G / (curr_iter + 1)
            avg_loss_D = all_loss_D / (curr_iter + 1)
            avg_loss_A = all_loss_A / (curr_iter + 1)

            writer.add_scalar('D_loss', value=errD.item(), step=curr_iter)
            writer.add_scalar('G_loss', value=errG.item(), step=curr_iter)
            writer.add_scalar('Acc', value=accuracy, step=curr_iter)

            print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.item(), avg_loss_D, errG.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))

        # 保存图像和checkpoint
        if epoch % opt.save_freq == 0 or epoch == 499:
            save_samples(real_cpu, path=os.path.join(opt.result_path, 'real_samples_epoch_{}.png'.format(epoch)))
            save_samples(fake,  path=os.path.join(opt.result_path, 'fake_samples_epoch_{}.png'.format(epoch)))
            paddle.save(netG.state_dict(), os.path.join(opt.check_path, 'G_{}.pdparams'.format(epoch)))
            paddle.save(netD.state_dict(), os.path.join(opt.check_path, 'D_{}.pdparams'.format(epoch)))
