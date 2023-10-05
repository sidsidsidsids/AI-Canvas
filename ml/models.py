import tensorflow as tf
import tensorflow_addons as tfa


class Residual_Block(tf.keras.Model):
    '''
    Residual Block 클래스:
        Conv2d - InstanceNorm - Relu - Conv2d - InstanceNorm - Add(Residual Connection) 으로 구성
        artifact를 줄이기 위해 reflection padding 채택
    '''

    def __init__(self, input_channels):
        super(Residual_Block, self).__init__()

        self.conv_1 = tf.keras.layers.Conv2D(filters=input_channels, kernel_size=3, padding='valid', use_bias=False,
                                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.instance_norm_1 = tfa.layers.InstanceNormalization()

        self.conv_2 = tf.keras.layers.Conv2D(filters=input_channels, kernel_size=3, padding='valid', use_bias=False,
                                             kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.instance_norm_2 = tfa.layers.InstanceNormalization()

        self.activation = tf.keras.layers.ReLU()

    def reflection_pad(self, input, pad_size):
        return tf.pad(input, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='REFLECT')

    def call(self, inputs):
        x = self.reflection_pad(inputs, 1)
        x = self.conv_1(x)
        x = self.instance_norm_1(x)
        x = self.activation(x)

        x = self.reflection_pad(x, 1)
        x = self.conv_2(x)
        x = self.instance_norm_2(x)

        return x + inputs
    

class Pix2Pix_Generator(tf.keras.Model):
    '''
    Pix2Pix Generator 클래스:
        contracting block + 9 residual blocks + expanding block 으로 구성 => CycleGAN에서의 generator
        (기존 U-net 기반 방식 대신 resnet 기반 방식을 채택)
    '''

    def __init__(self, input_channels, output_channels, hidden_channels=64, name=""):
        super(Pix2Pix_Generator, self).__init__()

        if name:
            self._name = name

        # CycleGAN 저자들의 notation 참고
        # for c7s1-64
        self.c7s1_64_conv = tf.keras.layers.Conv2D(filters=hidden_channels, kernel_size=7, padding='valid', use_bias=False,
                                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.c7s1_64_instance_norm = tfa.layers.InstanceNormalization()

        # for d128
        self.d128_conv = tf.keras.layers.Conv2D(filters=2*hidden_channels, kernel_size=3, strides=2, padding='valid', use_bias=False,
                                                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.d128_instance_norm = tfa.layers.InstanceNormalization()

        # for d256
        self.d256_conv = tf.keras.layers.Conv2D(filters=4*hidden_channels, kernel_size=3, strides=2, padding='valid', use_bias=False,
                                                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.d256_instance_norm = tfa.layers.InstanceNormalization()

        # for residual blocks
        R_channels = 4*hidden_channels
        self.R256_1 = Residual_Block(R_channels)
        self.R256_2 = Residual_Block(R_channels)
        self.R256_3 = Residual_Block(R_channels)
        self.R256_4 = Residual_Block(R_channels)
        self.R256_5 = Residual_Block(R_channels)
        self.R256_6 = Residual_Block(R_channels)
        self.R256_7 = Residual_Block(R_channels)
        self.R256_8 = Residual_Block(R_channels)
        self.R256_9 = Residual_Block(R_channels)

        # for u128
        self.u128_conv_transpose = tf.keras.layers.Conv2DTranspose(filters=2*hidden_channels, kernel_size=3, strides=2, padding='same', use_bias=False,
                                                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.u128_instance_norm = tfa.layers.InstanceNormalization()

        # for u256
        self.u256_conv_transpose = tf.keras.layers.Conv2DTranspose(filters=hidden_channels, kernel_size=3, strides=2, padding='same', use_bias=False,
                                                                   kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))
        self.u256_instance_norm = tfa.layers.InstanceNormalization()

        # for c7s1-3
        self.c7s1_3_conv = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=7, padding='valid',
                                                  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

        # activation
        self.relu = tf.keras.layers.ReLU()


    def reflection_pad(self, input, pad_size):
        return tf.pad(input, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], mode='REFLECT')


    def call(self, inputs):
        # c7s1-64
        x = self.reflection_pad(inputs, 3)
        x = self.c7s1_64_conv(x)
        x = self.c7s1_64_instance_norm(x)
        x = self.relu(x)

        # d128
        x = self.reflection_pad(x, 1)
        x = self.d128_conv(x)
        x = self.d128_instance_norm(x)
        x = self.relu(x)

        # d256
        x = self.reflection_pad(x, 1)
        x = self.d256_conv(x)
        x = self.d256_instance_norm(x)
        x = self.relu(x)

        # R256_1~9
        x = self.R256_1(x)
        x = self.R256_2(x)
        x = self.R256_3(x)
        x = self.R256_4(x)
        x = self.R256_5(x)
        x = self.R256_6(x)
        x = self.R256_7(x)
        x = self.R256_8(x)
        x = self.R256_9(x)

        # u128
        x = self.u128_conv_transpose(x)
        x = self.u128_instance_norm(x)
        x = self.relu(x)

        # u256
        x = self.u256_conv_transpose(x)
        x = self.u256_instance_norm(x)
        x = self.relu(x)

        # c7s1-3
        x = self.reflection_pad(x, 3)
        x = self.c7s1_3_conv(x)

        return tf.keras.activations.tanh(x)
