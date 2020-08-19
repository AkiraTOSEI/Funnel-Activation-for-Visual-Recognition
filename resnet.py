import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, Add, TimeDistributed, Flatten, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, GlobalMaxPooling3D
from tensorflow.keras.layers import Layer
from tensorflow_addons.layers import GroupNormalization

from FReLU import FReLU

def define_NormLayers(norm):
    if norm == "BatchNorm":
        return BatchNormalization
    elif norm == "GroupNorm":
        return GroupNormalization
    else:
        raise Exception(
            "Normalization that you specify is invalid! Current value:", norm)


def define_ConvLayer(mode):
    if mode == "2D" or mode == "TimeD":
        return Conv2D
    elif mode == "1D":
        return Conv1D
    elif mode == "3D":
        return Conv3D
    else:
        raise Exception(
            "Convolution mode that you specify is invalid! Current value:", mode)


def define_Pooling(mode):
    if mode == "2D" or mode == "TimeD":
        return MaxPooling2D
    elif mode == "1D":
        return MaxPooling1D
    elif mode == "3D":
        return MaxPooling3D
    else:
        raise Exception(
            "Convolution mode that you specify is invalid! Current value:", mode)


def define_activation(act):
    if act == 'ReLU':
        return Activation('relu')
    elif act == "Swish":
        return tf.keras.activations.swish
    elif act == 'FReLU':
        return FReLU()
    else:
        raise Exception(
            "Please use 'ReLU','Swish' or 'FReLU'. current value:", act)


def define_GlobalPooling(mode, pooling):
    if (mode == "2D" or mode == "TimeD") and pooling == "max":
        return GlobalMaxPooling2D
    elif mode == "1D" and pooling == "max":
        return GlobalMaxPooling1D
    elif mode == "3D" and pooling == "max":
        return GlobalMaxPooling3D
    elif (mode == "2D" or mode == "TimeD") and pooling == "ave":
        return GlobalAveragePooling2D
    elif mode == "1D" and pooling == "ave":
        return GlobalAveragePooling1D
    elif mode == "3D" and pooling == "ave":
        return GlobalAveragePooling3D


def Conv_stage1_block(x, filters, strides=2, mode="2D", norm="BatchNorm", act='ReLU', kernel_initializer='glorot_uniform', name=None):
    NormLayer = define_NormLayers(norm)  # Define Normalization Layers
    ConvLayer = define_ConvLayer(mode)  # Define ConvLayer
    MaxPooling = define_Pooling(mode)  # Define Pooling
    if mode == "1D" or mode == "2D" or mode == "3D":
        conv1 = ConvLayer(filters, kernel_size=7, strides=strides,
                          kernel_initializer=kernel_initializer, padding='same')
        bn1 = NormLayer()
        act1 = define_activation(act)
        pool1 = MaxPooling(pool_size=3, strides=2)
    elif mode == "TimeD":
        conv1 = TimeDistributed(ConvLayer(
            filters, kernel_size=7, kernel_initializer=kernel_initializer, strides=strides, padding='same'))
        bn1 = TimeDistributed(NormLayer())
        act1 = TimeDistributed(define_activation(act))
        pool1 = TimeDistributed(MaxPooling(pool_size=(3, 3), strides=(2, 2)))

    h = conv1(x)
    h = bn1(h)
    h = act1(h)
    output = pool1(h)
    return output


def Identity_bottleneck_block(x, filters, kernel_size=3, mode="2D", norm="BatchNorm", act='ReLU', kernel_initializer='glorot_uniform', name=None):
    """A block that has no conv layer at shortcut for ResNet50, ResNet101,ResNet152.
    # Arguments
        filters(list): list of integers, the filters of 3 conv layer at main path
        kernel_size(int): default 3, the kernel size of middle conv layer at main path
        strides(int or list): default 2, the stride size of middle conv layer at main path
        mode(str):Conv1D("1D"), Conv2D("2D"),Conv3D("3D"),TimedistributedConv2D("TimeD")
        norm(str): Normalization option.BatchNormalization("BatchNorm") or GroupNormaliztion("GroupNorm")
        kernel_initializer(str):kernel_initializer of Convolutional Layer

    # Returns
        Output tensor for the block.
    """
    NormLayer = define_NormLayers(norm)  # Define Normalization Layers
    ConvLayer = define_ConvLayer(mode)

    filters1, filters2, filters3 = filters
    if mode == "1D" or mode == "2D" or mode == "3D":
        conv1 = ConvLayer(filters1, 1, kernel_initializer=kernel_initializer)
        bn1 = NormLayer()
        relu1 = define_activation(act)
        conv2 = ConvLayer(filters2,  kernel_size,
                          kernel_initializer=kernel_initializer, padding='same')
        bn2 = NormLayer()
        relu2 = define_activation(act)
        conv3 = ConvLayer(filters3, 1, kernel_initializer=kernel_initializer)
        bn3 = NormLayer()
        relu_m = define_activation(act)
    elif mode == "TimeD":
        conv1 = TimeDistributed(ConvLayer(
            filters1, (1, 1), kernel_initializer=kernel_initializer, padding='same'))
        bn1 = TimeDistributed(NormLayer())
        relu1 = TimeDistributed(define_activation(act))
        conv2 = TimeDistributed(ConvLayer(
            filters2,  kernel_size, kernel_initializer=kernel_initializer, padding='same'))
        bn2 = TimeDistributed(NormLayer())
        relu2 = TimeDistributed(define_activation(act))
        conv3 = TimeDistributed(ConvLayer(
            filters3, (1, 1), kernel_initializer=kernel_initializer, padding='same'))
        bn3 = TimeDistributed(NormLayer())
        relu_m = TimeDistributed(define_activation(act))

    residual = x
    h = conv1(x)
    h = bn1(h)
    h = relu1(h)
    h = conv2(h)
    h = bn2(h)
    h = relu2(h)
    h = conv3(h)
    h = bn3(h)
    # Merge
    output = Add()([residual, h])
    output = relu_m(output)
    return output


def Conv_bottleneck_block(x, filters, kernel_size=3, strides=2, mode="2D", norm="BatchNorm", act='ReLU', kernel_initializer='glorot_uniform', name=None):
    """A block that has conv layer at shortcut for ResNet50, ResNet101,ResNet152.
    # Arguments
        filters(list): list of integers, the filters of 3 conv layer at main path
        kernel_size(int): default 3, the kernel size of middle conv layer at main path
        strides(int or list): default 2, the stride size of middle conv layer at main path
        mode(str):Conv1D("1D"), Conv2D("2D"),Conv3D("3D"),TimedistributedConv2D("TimeD")
        norm(str): Normalization option.BatchNormalization("BatchNorm") or GroupNormaliztion("GroupNorm")
        kernel_initializer(str):kernel_initializer of Convolutional Layer

    # Returns
        Output tensor for the block.
    """
    NormLayer = define_NormLayers(norm)  # Define Normalization Layers
    ConvLayer = define_ConvLayer(mode)  # Define ConvLayer

    filters1, filters2, filters3 = filters
    if mode == "1D" or mode == "2D" or mode == "3D":
        # Main Path
        bn1 = NormLayer()
        relu1 = define_activation(act)
        conv1 = ConvLayer(filters1, 1, strides=strides,
                          kernel_initializer=kernel_initializer)
        bn2 = NormLayer()
        relu2 = define_activation(act)
        conv2 = ConvLayer(filters2,  kernel_size,
                          kernel_initializer=kernel_initializer, padding='same')
        bn3 = NormLayer()
        conv3 = ConvLayer(filters3, 1, kernel_initializer=kernel_initializer)
        # Short cut path
        s_bn = NormLayer()
        s_conv = ConvLayer(filters3, 1, strides=strides,
                           kernel_initializer=kernel_initializer)
        relu_m = define_activation(act)

    elif mode == "TimeD":
        # Main Path
        bn1 = TimeDistributed(NormLayer())
        relu1 = TimeDistributed(define_activation(act))
        conv1 = TimeDistributed(ConvLayer(
            filters1, (1, 1), strides=strides, kernel_initializer=kernel_initializer))
        bn2 = TimeDistributed(NormLayer())
        relu2 = TimeDistributed(define_activation(act))
        conv2 = TimeDistributed(ConvLayer(
            filters2,  kernel_size, kernel_initializer=kernel_initializer, padding='same'))
        bn3 = TimeDistributed(NormLayer())
        conv3 = TimeDistributed(
            ConvLayer(filters3, (1, 1), kernel_initializer=kernel_initializer))
        # Short cut path
        s_bn = TimeDistributed(NormLayer())
        s_conv = TimeDistributed(ConvLayer(
            filters3, (1, 1), strides=strides, kernel_initializer=kernel_initializer))
        # merge
        relu_m = TimeDistributed(define_activation(act))

    residual = x
    # Main Path
    h = conv1(x)
    h = bn1(h)
    h = relu1(h)
    h = conv2(h)
    h = bn2(h)
    h = relu2(h)
    h = conv3(h)
    h = bn3(h)
    # Short cut path
    residual = s_conv(residual)
    residual = s_bn(residual)
    # Merge
    output = Add()([residual, h])
    output = relu_m(output)
    return output


def Identity_basic_block(x, filters, kernel_size=3,  mode="2D", norm="BatchNorm", act='ReLU', kernel_initializer='glorot_uniform', name=None):
    """A block that has no conv layer at shortcut for ResNet18, ResNet34.
    # Arguments
        filters(list): list of integers, the filters of 2 conv layer at main path
        kernel_size(int): default 3, the kernel size of middle conv layer at main path
        strides(int or list): default 2, the stride size of middle conv layer at main path
        mode(str):Conv1D("1D"), Conv2D("2D"),Conv3D("3D"),TimedistributedConv2D("TimeD")
        norm(str): Normalization option.BatchNormalization("BatchNorm") or GroupNormaliztion("GroupNorm")
        kernel_initializer(str):kernel_initializer of Convolutional Layer

    # Returns
        Output tensor for the block.
    """
    NormLayer = define_NormLayers(norm)  # Define Normalization Layers
    ConvLayer = define_ConvLayer(mode)  # Define ConvLayer

    filters1, filters2 = filters
    if mode == "1D" or mode == "2D" or mode == "3D":
        conv1 = ConvLayer(filters1, kernel_size,
                          kernel_initializer=kernel_initializer, padding='same')
        bn1 = NormLayer()
        relu1 = define_activation(act)
        conv2 = ConvLayer(filters2,  kernel_size,
                          kernel_initializer=kernel_initializer, padding='same')
        bn2 = NormLayer()
        relu_m = define_activation(act)
    elif mode == "TimeD":
        conv1 = TimeDistributed(ConvLayer(
            filters1, kernel_size, kernel_initializer=kernel_initializer, padding='same'))
        bn1 = TimeDistributed(NormLayer())
        relu1 = TimeDistributed(define_activation(act))
        conv2 = TimeDistributed(ConvLayer(
            filters2,  kernel_size, kernel_initializer=kernel_initializer, padding='same'))
        bn2 = TimeDistributed(NormLayer())
        relu_m = TimeDistributed(define_activation(act))

    residual = x
    h = conv1(x)
    h = bn1(h)
    h = relu1(h)
    h = conv2(h)
    h = bn2(h)
    # Merge
    output = Add()([residual, h])
    output = relu_m(output)
    return output


def Conv_basic_block(x, filters, kernel_size=3, strides=2, mode="2D", norm="BatchNorm", act='ReLU', kernel_initializer='glorot_uniform', name=None):
    """A block that has a conv layer at shortcut for ResNet18, ResNet34.
    # Arguments
        filters(list): list of integers, the filters of 2 conv layer at main path
        kernel_size(int): default 3, the kernel size of middle conv layer at main path
        strides(int or list): default 2, the stride size of middle conv layer at main path
        mode(str):Conv1D("1D"), Conv2D("2D"),Conv3D("3D"),TimedistributedConv2D("TimeD")
        norm(str): Normalization option.BatchNormalization("BatchNorm") or GroupNormaliztion("GroupNorm")
        kernel_initializer(str):kernel_initializer of Convolutional Layer

    # Returns
        Output tensor for the block.
    """
    NormLayer = define_NormLayers(norm)  # Define Normalization Layers
    ConvLayer = define_ConvLayer(mode)  # Define ConvLayer

    filters1, filters2 = filters
    if mode == "1D" or mode == "2D" or mode == "3D":
        # Main Path
        conv1 = ConvLayer(filters1, 1, strides=strides,
                          kernel_initializer=kernel_initializer, padding='same')
        bn1 = NormLayer()
        relu1 = define_activation(act)
        bn2 = NormLayer()
        conv2 = ConvLayer(filters2,  kernel_size,
                          kernel_initializer=kernel_initializer, padding='same')
        # Short cut path
        s_bn = NormLayer()
        s_conv = ConvLayer(filters2, 1, strides=strides,
                           kernel_initializer=kernel_initializer, padding='same')
        # merge
        relu_m = define_activation(act)
    elif mode == "TimeD":
        # Main Path
        conv1 = TimeDistributed(ConvLayer(
            filters1, (1, 1), strides=strides, kernel_initializer=kernel_initializer, padding='same'))
        bn1 = TimeDistributed(NormLayer())
        relu1 = TimeDistributed(define_activation(act))
        conv2 = TimeDistributed(ConvLayer(
            filters2,  kernel_size, kernel_initializer=kernel_initializer, padding='same'))
        bn2 = TimeDistributed(NormLayer())
        # Short cut path
        s_bn = TimeDistributed(NormLayer())
        s_conv = TimeDistributed(ConvLayer(
            filters2, (1, 1), strides=strides, kernel_initializer=kernel_initializer, padding='same'))
        # merge
        relu_m = TimeDistributed(define_activation(act))

    residual = x
    # Main Path
    h = conv1(x)
    h = bn1(h)
    h = relu1(h)
    h = conv2(h)
    h = bn2(h)
    # Short cut path
    residual = s_conv(residual)
    residual = s_bn(residual)
    # Merge
    output = Add()([residual, h])
    output = relu_m(output)
    return output


def Fin_layer(x, mode="2D", classes=1000, include_top=True, pooling=None, name=None):

    GlobalPooling = define_GlobalPooling(mode, pooling)
    if mode == "1D" or mode == "2D" or mode == "3D":
        # Pooling setting
        gp = GlobalPooling()
        if include_top:
            dense = Dense(classes, 'softmax')
    elif mode == "TimeD":
        gp = TimeDistributed(GlobalPooling())
        if include_top:
            flat = Flatten()
            dense = Dense(classes, 'softmax')

    output = gp(x)
    if include_top and (mode == "1D" or mode == "2D" or mode == "3D"):
        output = dense(output)
    if include_top and mode == "TimeD":
        output = flat(output)
        output = dense(output)
    return output


class ResnetBuilder():
    '''ResNet builder
    '''

    def __init__(self, name, classes=1000, include_top=True, pooling=None, mode="2D", norm="BatchNorm", act='ReLU', kernel_initializer='glorot_uniform', seqence_length=None, input_shape=(224, 224, 3)):
        '''
        this model has similar augments to tf.keras.application.ResNet50 
        Args:
            include_top(bool): whether to include the fully-connected layer at the top of the network.
            pooling(bool): Optional pooling mode for feature extraction when include_top is False.
            classes(int): optional number of classes to classify images into, only to be specified if include_top is True
            norm(str): Normalization option.BatchNormalization("BatchNorm") or GroupNormaliztion("GroupNorm")
            kernel_initializer(str):kernel_initializer of Convolutional Layer
            mode(str):Conv1D("1D"), Conv2D("2D"),Conv3D("3D"),TimedistributedConv2D("TimeD")
            seqence_length(int) : sequence length of time dimension when using Conv3D.
        '''

        if not (mode == "1D" or mode == "2D" or mode == "TimeD" or mode == "3D"):
            raise Exception(
                "'mode' value is invalid. you should use '1D' or '2D' or '3D' or 'TimeD'. Current value :", mode)
        if not (pooling == "ave" or pooling == "max" or pooling == None):
            raise Exception(
                "'pooling' value is invalid. you should use 'ave' or 'max' or None. Current value :", pooling)
        if not (include_top == True or include_top == False):
            raise Exception(
                "'include_top' value is invalid. you should use bool value. Current value :", include_top)
        self.include_top = include_top
        if pooling == None and include_top == True:
            self.pooling = "ave"
        else:
            self.pooling = pooling

        self.norm = norm
        self.mode = mode
        self.act = act
        self.kernel_initializer = kernel_initializer
        self.classes = classes
        self.input_shape = input_shape

        # name setting
        if name.startswith('WS'):
            wide = int(name[2])
            resnet_name = name[3:]
            stage_filters = np.array([64, 128, 256, 512])*wide
            self.stage_filters = list(stage_filters)

        elif name.startswith('GMPWS'):
            wide = int(name[5])
            resnet_name = name[6:]
            stage_filters = np.array([64, 128, 256, 512])*wide
            self.stage_filters = list(stage_filters)
            self.pooling = 'max'

        elif name.startswith('ResNet'):
            self.stage_filters = [64, 128, 256, 512]
            resnet_name = name
        else:
            raise Exception('name error : ', name)

        if resnet_name == "ResNet18":
            self.block_type = "basic"
            self.reptitions = [2, 2, 2, 2]
        elif resnet_name == "ResNet34":
            self.block_type = "basic"
            self.reptitions = [3, 4, 6, 3]
        elif resnet_name == "ResNet50":
            self.block_type = "bottleneck"
            self.reptitions = [3, 4, 6, 3]
        elif resnet_name == "ResNet101":
            self.block_type = "bottleneck"
            self.reptitions = [3, 4, 23, 3]
        elif resnet_name == "ResNet152":
            self.block_type = "bottleneck"
            self.reptitions = [3, 8, 36, 3]
        else:
            raise Exception(
                " Name Error! you can use ResNet18,ResNet34,ResNet50,ResNet101, or ResNet152. Current name:", name)

        if seqence_length != None and mode == "3D":
            # ajustment of sequence length of time dimension.
            # usualy time length is smaller than image size. So need to ajust the number of
            # delete dimension of time.
            if 32 <= seqence_length:
                self.strides_list = [[2, 2, 2], [
                    2, 2, 2], [2, 2, 2], [2, 2, 2]]
            elif 16 <= seqence_length and seqence_length < 32:
                self.strides_list = [[2, 2, 2], [
                    2, 2, 2], [2, 2, 2], [1, 2, 2]]
            elif 8 <= seqence_length and seqence_length < 16:
                self.strides_list = [[2, 2, 2], [
                    2, 2, 2], [1, 2, 2], [1, 2, 2]]
            elif 4 <= seqence_length and seqence_length < 8:
                self.strides_list = [[2, 2, 2], [
                    1, 2, 2], [1, 2, 2], [1, 2, 2]]
            elif 2 <= seqence_length and seqence_length < 4:
                self.strides_list = [[1, 2, 2], [
                    1, 2, 2], [1, 2, 2], [1, 2, 2]]
        else:
            self.strides_list = [2, 2, 2, 2]

        self.define_block_type()

    def define_block_type(self):
        '''define block type 
        '''
        if self.block_type == "basic":
            self.IdBlock = Identity_basic_block
            self.ConvBlock = Conv_basic_block
            self.all_filters = []
            for s_f in self.stage_filters:
                self.all_filters.append([s_f, s_f])

        elif self.block_type == "bottleneck":
            self.IdBlock = Identity_bottleneck_block
            self.ConvBlock = Conv_bottleneck_block
            self.all_filters = []
            for s_f in self.stage_filters:
                self.all_filters.append([s_f, s_f, s_f*4])

    def builder(self):
        input_layer = Input(self.input_shape)
        # stage1 (Use ReLU in 1st stage, not using FReLU.)
        h = Conv_stage1_block(
            input_layer,
            filters=self.all_filters[0][0],
            mode=self.mode,
            strides=self.strides_list[0],
            norm=self.norm,
            kernel_initializer=self.kernel_initializer
        )

        # stage2
        h = self.ConvBlock(h, filters=self.all_filters[0], strides=1, mode=self.mode,
                           norm=self.norm, act=self.act, kernel_initializer=self.kernel_initializer)
        for rep in range(1, self.reptitions[0]):
            h = self.IdBlock(h, filters=self.all_filters[0],
                             mode=self.mode, norm=self.norm, act=self.act, kernel_initializer=self.kernel_initializer)

        # stage3
        h = self.ConvBlock(h, filters=self.all_filters[1], strides=self.strides_list[1],
                           mode=self.mode, norm=self.norm, act=self.act, kernel_initializer=self.kernel_initializer)
        for rep in range(1, self.reptitions[1]):
            h = self.IdBlock(h, filters=self.all_filters[1],
                             mode=self.mode, norm=self.norm, act=self.act, kernel_initializer=self.kernel_initializer)
        # stage4
        h = self.ConvBlock(h, filters=self.all_filters[2], strides=self.strides_list[2],
                           mode=self.mode, norm=self.norm, act=self.act, kernel_initializer=self.kernel_initializer)
        for rep in range(1, self.reptitions[2]):
            h = self.IdBlock(h, filters=self.all_filters[2],
                             mode=self.mode, norm=self.norm, act=self.act, kernel_initializer=self.kernel_initializer)

        # stage5
        h = self.ConvBlock(h, filters=self.all_filters[3], strides=self.strides_list[3],
                           mode=self.mode, norm=self.norm, act=self.act, kernel_initializer=self.kernel_initializer)
        for rep in range(1, self.reptitions[3]):
            h = self.IdBlock(h, filters=self.all_filters[3],
                             mode=self.mode, norm=self.norm, act=self.act, kernel_initializer=self.kernel_initializer)

        # Final stage
        if self.include_top == True or (self.include_top == False and self.pooling != None):
            output = Fin_layer(h, mode=self.mode, include_top=self.include_top,
                               classes=self.classes, pooling=self.pooling)
            return tf.keras.Model(inputs=input_layer, outputs=output)
        else:
            return tf.keras.Model(inputs=input_layer, outputs=h)
