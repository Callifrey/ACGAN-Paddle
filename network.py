import paddle
import paddle.nn as nn


class Generator(nn.Layer):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(110, 768)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.Conv2DTranspose(768, 384, 4, 2, 0),
            nn.BatchNorm2D(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.Conv2DTranspose(384, 256, 4, 2, 1),
            nn.BatchNorm2D(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.Conv2DTranspose(256, 192, 4, 2, 1),
            nn.BatchNorm2D(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.Conv2DTranspose(192, 64, 4, 2, 1),
            nn.BatchNorm2D(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 6
        self.tconv6 = nn.Sequential(
            nn.Conv2DTranspose(64, 32, 4, 2, 1),
            nn.BatchNorm2D(32),
            nn.ReLU(True),
        )
        # Transposed Convolution 7
        self.tconv7 = nn.Sequential(
            nn.Conv2DTranspose(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, input):
        input = paddle.reshape(input, shape=[-1, self.nz])
        fc1 = self.fc1(input)
        fc1 = paddle.reshape(fc1, shape=[-1, 768, 1, 1])
        tconv2 = self.tconv2(fc1)
        tconv3 = self.tconv3(tconv2)
        tconv4 = self.tconv4(tconv3)
        tconv5 = self.tconv5(tconv4)
        tconv6 = self.tconv6(tconv5)
        tconv7 = self.tconv7(tconv6)
        output = tconv7
        return output


class Discriminator(nn.Layer):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2D(3, 16, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2D(16, 32, 3, 1, 1),
            nn.BatchNorm2D(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2D(32, 64, 3, 2, 1),
            nn.BatchNorm2D(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2D(64, 128, 3, 1, 1),
            nn.BatchNorm2D(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2D(128, 256, 3, 2, 1),
            nn.BatchNorm2D(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2D(256, 512, 3, 1, 1),
            nn.BatchNorm2D(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(16*16*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(16*16*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        flat6 = paddle.reshape(conv6, [-1, 16*16*512])
        fc_dis = self.fc_dis(flat6)
        fc_aux = self.fc_aux(flat6)
        classes = self.softmax(fc_aux)
        realfake = paddle.reshape(self.sigmoid(fc_dis), [-1, 1]).squeeze(1)
        return realfake, classes


if __name__ == '__main__':
    # test Generator
    noise = paddle.randn((1, 110, 1, 1))
    G = Generator(nz=110)
    img = G(noise)
    print(img.shape)  # [1,3,128,128]

    # test Discriminator
    D = Discriminator(num_classes=10)
    real_fake, classes = D(img)
    print(real_fake.shape) # [1,100]
    print(classes.shape)  #[1,10]
