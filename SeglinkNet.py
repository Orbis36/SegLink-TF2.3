import tensorflow as tf
import numpy as np

class SegLinkNet(tf.keras.Model):

    '''
    输入的应该是batch，构建整个网络的前向传播
    '''

    def __init__(self,Pretrain = False, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.Base_net = 'vgg'
        self.weight_initializer = tf.keras.initializers.he_normal()
        self.bias_initializer = tf.keras.initializers.zeros()
        self.Feature_layers = {"conv4_3": (64, 64), "fc7": (32, 32), "conv6_2": (16, 16),
                               "conv7_2": (8, 8), "conv8_2": (4, 4), "conv9_2": (2, 2)}
        self.weight_regularizer = tf.keras.regularizers.L2(0.0005)
        self.ModelInput = (512,512,3)
        self.Softmax = tf.keras.layers.Softmax()
        if Pretrain:
            self.ModelPath = Pretrain

    def build(self, shape = None):
        weight = self.weight_initializer
        bias = self.bias_initializer
        with tf.name_scope('vgg'):
            #Block1;InputSize = (512,512)
            self.conv1_1 = tf.keras.layers.Conv2D(64, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu',kernel_regularizer=self.weight_regularizer, name='conv1_1')
            self.conv1_2 = tf.keras.layers.Conv2D(64, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu',kernel_regularizer=self.weight_regularizer, name='conv1_2')
            self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same', name='Pool1')

            # Block2;InputSize = (256,256)
            self.conv2_1 = tf.keras.layers.Conv2D(128, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu',kernel_regularizer=self.weight_regularizer, name='conv2_1')
            self.conv2_2 = tf.keras.layers.Conv2D(128, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu',kernel_regularizer=self.weight_regularizer, name='conv2_2')
            self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same', name='Pool2')

            # Block3;InputSize = (128,128)
            self.conv3_1 = tf.keras.layers.Conv2D(256, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu',kernel_regularizer=self.weight_regularizer, name='conv3_1')
            self.conv3_2 = tf.keras.layers.Conv2D(256, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu',kernel_regularizer=self.weight_regularizer, name='conv3_2')
            self.conv3_3 = tf.keras.layers.Conv2D(256, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu',kernel_regularizer=self.weight_regularizer, name='conv3_3')
            self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same', name='Pool3')

            # Block4;InputSize = (64,64)
            self.conv4_1 = tf.keras.layers.Conv2D(512, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu',kernel_regularizer=self.weight_regularizer, name='conv4_1')
            self.conv4_2 = tf.keras.layers.Conv2D(512, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu',kernel_regularizer=self.weight_regularizer, name='conv4_2')
            self.conv4_3 = tf.keras.layers.Conv2D(512, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu',kernel_regularizer=self.weight_regularizer, name='conv4_3')
            self.pool4 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same', name='Pool4')

            # Block5;InputSize = (32,32)
            self.conv5_1 = tf.keras.layers.Conv2D(512, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu', kernel_regularizer=self.weight_regularizer, name='conv5_1')
            self.conv5_2 = tf.keras.layers.Conv2D(512, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu', kernel_regularizer=self.weight_regularizer, name='conv5_2')
            self.conv5_3 = tf.keras.layers.Conv2D(512, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                                  activation='relu', kernel_regularizer=self.weight_regularizer, name='conv5_3')

            # fc6 as conv, dilation is added
            self.fc6 = tf.keras.layers.Conv2D(1024, 3, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                              activation='relu', kernel_regularizer=self.weight_regularizer, name='fc6',
                                              dilation_rate=6)
            # fc7 as conv
            self.fc7 = tf.keras.layers.Conv2D(1024, 1, 1, padding='same', kernel_initializer=weight, bias_initializer=bias,
                                              activation='relu', kernel_regularizer=self.weight_regularizer, name='fc7')
            #here size is (32,32)
        with tf.name_scope('extra_layers'):
            self.conv6_1 = tf.keras.layers.Conv2D(256, 1, 1, padding='same',kernel_initializer=self.weight_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.weight_regularizer,
                                                  activation='relu',name='conv6_1')
            self.conv6_2 = tf.keras.layers.Conv2D(512, 3, 2, padding='same',kernel_initializer=self.weight_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.weight_regularizer,
                                                  activation='relu',name='conv6_2')
            self.conv7_1 = tf.keras.layers.Conv2D(128, 1, 1, padding='same',kernel_initializer=self.weight_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.weight_regularizer,
                                                  activation='relu', name='conv7_1')
            self.conv7_2 = tf.keras.layers.Conv2D(256, 3, 2, padding='same',kernel_initializer=self.weight_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.weight_regularizer,
                                                  activation='relu', name='conv7_2')
            self.conv8_1 = tf.keras.layers.Conv2D(128, 1, 1, padding='same',kernel_initializer=self.weight_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.weight_regularizer,
                                                  activation='relu', name='conv8_1')
            self.conv8_2 = tf.keras.layers.Conv2D(256, 3, 2, padding='same',kernel_initializer=self.weight_initializer,
                                                  bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.weight_regularizer,
                                                  activation='relu', name='conv8_2')
            self.conv9_1 = tf.keras.layers.Conv2D(128, 1, 1, padding='same',kernel_initializer=self.weight_initializer
                                                  , bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.weight_regularizer,
                                                  activation='relu', name='conv9_1')
            self.conv9_2 = tf.keras.layers.Conv2D(256, 3, 2, padding='same',
                                                  kernel_initializer=self.weight_initializer
                                                  , bias_initializer=self.bias_initializer,
                                                  kernel_regularizer=self.weight_regularizer,
                                                  activation='relu', name='conv9_2')
        with tf.name_scope('seglink_layers'):
            num_cls_pred = 2
            num_offset_pred = 5
            num_cross_layer_link_scores_pred = 8
            num_within_layer_link_scores_pred = 16

            with tf.name_scope('conv4_3'):
                self.conv4_3CP_num_cls_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(64, 64, 512)),
                    tf.keras.layers.Conv2D(num_cls_pred, 3, padding='same', name='num_cls_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv4_3CP_num_offset_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(64, 64, 512)),
                    tf.keras.layers.Conv2D(num_offset_pred, 3, padding='same', name='num_offset_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv4_3CP_num_within_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(64, 64, 512)),
                    tf.keras.layers.Conv2D(num_within_layer_link_scores_pred, 3, padding='same',
                                           name='within_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
            with tf.name_scope('fc7'):
                self.FC7_num_cls_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(32, 32, 1024)),
                    tf.keras.layers.Conv2D(num_cls_pred, 3, padding='same', name='num_cls_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.FC7_num_offset_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(32, 32, 1024)),
                    tf.keras.layers.Conv2D(num_offset_pred, 3, padding='same', name='num_offset_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.FC7_num_cross_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(32, 32, 1024)),
                    tf.keras.layers.Conv2D(num_cross_layer_link_scores_pred, 3, padding='same',
                                           name='cross_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.FC7_num_within_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(32, 32, 1024)),
                    tf.keras.layers.Conv2D(num_within_layer_link_scores_pred, 3, padding='same',
                                           name='within_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
            with tf.name_scope('conv6_2'):
                self.conv6_2CP_num_cls_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(16, 16, 512)),
                    tf.keras.layers.Conv2D(num_cls_pred, 3, padding='same', name='num_cls_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv6_2CP_num_offset_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(16, 16, 512)),
                    tf.keras.layers.Conv2D(num_offset_pred, 3, padding='same', name='num_offset_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv6_2CP_num_cross_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(16, 16, 512)),
                    tf.keras.layers.Conv2D(num_cross_layer_link_scores_pred, 3, padding='same',
                                           name='cross_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv6_2CP_num_within_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(16, 16, 512)),
                    tf.keras.layers.Conv2D(num_within_layer_link_scores_pred, 3, padding='same',
                                           name='within_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
            with tf.name_scope('conv7_2'):
                self.conv7_2CP_num_cls_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(8, 8, 256)),
                    tf.keras.layers.Conv2D(num_cls_pred, 3, padding='same', name='num_cls_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv7_2CP_num_offset_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(8, 8, 256)),
                    tf.keras.layers.Conv2D(num_offset_pred, 3, padding='same', name='num_offset_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv7_2CP_num_cross_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(8, 8, 256)),
                    tf.keras.layers.Conv2D(num_cross_layer_link_scores_pred, 3, padding='same',
                                           name='cross_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv7_2CP_num_within_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(8, 8, 256)),
                    tf.keras.layers.Conv2D(num_within_layer_link_scores_pred, 3, padding='same',
                                           name='within_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
            with tf.name_scope('conv8_2'):
                self.conv8_2CP_num_cls_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(4, 4, 256)),
                    tf.keras.layers.Conv2D(num_cls_pred, 3, padding='same', name='num_cls_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv8_2CP_num_offset_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(4, 4, 256)),
                    tf.keras.layers.Conv2D(num_offset_pred, 3, padding='same', name='num_offset_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv8_2CP_num_cross_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(4, 4, 256)),
                    tf.keras.layers.Conv2D(num_cross_layer_link_scores_pred, 3, padding='same',
                                           name='cross_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv8_2CP_num_within_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(4, 4, 256)),
                    tf.keras.layers.Conv2D(num_within_layer_link_scores_pred, 3, padding='same',
                                           name='within_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
            with tf.name_scope('conv9_2'):
                self.conv9_2CP_num_cls_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(2, 2, 256)),
                    tf.keras.layers.Conv2D(num_cls_pred, 3, padding='same', name='num_cls_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv9_2CP_num_offset_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(2, 2, 256)),
                    tf.keras.layers.Conv2D(num_offset_pred, 3, padding='same', name='num_offset_pred',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv9_2CP_num_cross_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(2, 2, 256)),
                    tf.keras.layers.Conv2D(num_cross_layer_link_scores_pred, 3, padding='same',
                                           name='cross_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])
                self.conv9_2CP_num_within_layer_link_scores_pred = tf.keras.Sequential([
                    tf.keras.layers.InputLayer(input_shape=(2, 2, 256)),
                    tf.keras.layers.Conv2D(num_within_layer_link_scores_pred, 3, padding='same',
                                           name='within_layer_link_scores',
                                           kernel_initializer=self.weight_initializer,
                                           kernel_regularizer=self.weight_regularizer)
                ])

    def call(self, *args, **kwargs):
        try:
            self.ModelPath
        except AttributeError:
            print("NO PreTrain Model Exist, Train from 0")
        else:

            fake_img = tf.expand_dims(np.zeros(self.ModelInput),0)
            Endpoint = self.GetEndPoint(fake_img)
            self._add_seglink_estimator(Endpoint)
            self.load_weights(self.ModelPath)
            print("Load Parameters successfully")

    def GetEndPoint(self,inputs):

        def VGGmodel1(inputs):
            x = self.conv1_1(inputs)
            x = self.conv1_2(x)
            x = self.pool1(x)
            x = self.conv2_1(x)
            x = self.conv2_2(x)
            x = self.pool2(x)
            x = self.conv3_1(x)
            x = self.conv3_2(x)
            x = self.conv3_3(x)
            x = self.pool3(x)
            x = self.conv4_1(x)
            x = self.conv4_2(x)
            x = self.conv4_3(x)

            return x
        def VGGmodel2(inputs):
            x = self.pool4(inputs)
            x = self.conv5_1(x)
            x = self.conv5_2(x)
            x = self.conv5_3(x)
            x = self.fc6(x)
            x = self.fc7(x)

            return x
        def Sub_model1(inputs):
            x = self.conv6_1(inputs)
            x = self.conv6_2(x)
            return x
        def Sub_model2(inputs):
            x = self.conv7_1(inputs)
            x = self.conv7_2(x)
            return x
        def Sub_model3(inputs):
            x = self.conv8_1(inputs)
            x = self.conv8_2(x)
            return x
        def Sub_model4(inputs):
            x = self.conv9_1(inputs)
            x = self.conv9_2(x)
            return x

        Endpoint = self.Feature_layers  # key值一样，简单做个代替
        Endpoint['conv4_3'] = VGGmodel1(inputs)#(64,64,512)
        Endpoint['fc7'] = VGGmodel2(Endpoint['conv4_3'])#(32,32,1024)
        Endpoint['conv6_2'] = Sub_model1(Endpoint['fc7'])#(16,16,512)
        Endpoint['conv7_2'] = Sub_model2(Endpoint['conv6_2'])#(8,8,256)
        Endpoint['conv8_2'] = Sub_model3(Endpoint['conv7_2'])#(4,4,256)
        Endpoint['conv9_2'] = Sub_model4(Endpoint['conv8_2'])#(2,2,256)
        return Endpoint

    def _add_seglink_estimator(self,End_point):

        self.End_point = End_point

        all_seg_scores = []
        all_seg_offsets = []
        all_within_layer_link_scores = []
        all_cross_layer_link_scores = []

        for layer_name in self.Feature_layers:
            seg_scores, seg_offsets, within_layer_link_scores, cross_layer_link_scores = \
                self._build_seg_link_layer(layer_name)
            #print(seg_offsets.shape)

            all_seg_scores.append(seg_scores)
            all_seg_offsets.append(seg_offsets)
            all_within_layer_link_scores.append(within_layer_link_scores)
            all_cross_layer_link_scores.append(cross_layer_link_scores)

        def reshape_and_concat(tensors):
            def reshape(t):
                shape = t.get_shape().as_list()
                if len(shape) == 4:
                    shape = (shape[0], -1, shape[-1])  # [x,y,4,2]->[x,-1,2]
                    t = tf.reshape(t, shape)
                elif len(shape) == 5:
                    shape = (shape[0], -1, shape[-2], shape[-1])
                    t = tf.reshape(t, shape)
                    t = tf.reshape(t, [shape[0], -1, shape[-1]])
                else:
                    raise ValueError("invalid tensor shape: %s, shape = %s" % (t.name, shape))
                return t

            reshaped_tensors = [reshape(t) for t in tensors if t is not None]
            return tf.concat(reshaped_tensors, axis=1)


        seg_score_logits = reshape_and_concat(all_seg_scores)  # (batch_size, N, 2)
        seg_offsets = reshape_and_concat(all_seg_offsets)  # (batch_size, N, 5)
        cross_layer_link_scores = reshape_and_concat(all_cross_layer_link_scores)  # (batch_size, 8N, 2)
        within_layer_link_scores = reshape_and_concat(
            all_within_layer_link_scores)  # (batch_size, 4(N - N_conv4_3), 2)

        link_score_logits = tf.concat([within_layer_link_scores, cross_layer_link_scores], axis=1)
        seg_scores = self.Softmax(seg_score_logits)# (batch_size, N, 2)
        link_scores = self.Softmax(link_score_logits)

        return seg_score_logits, seg_offsets, seg_scores, link_scores, link_score_logits

    def _build_seg_link_layer(self, layer_name):
        Featurelayer = self.End_point[layer_name]
        if layer_name == 'conv4_3':
            Featurelayer = tf.nn.l2_normalize(Featurelayer,-1,name='conv4_3Norm')*20
            seg_scores = self.conv4_3CP_num_cls_pred(Featurelayer)
            seg_offsets = self.conv4_3CP_num_offset_pred(Featurelayer)
            within_layer_link_scores = self.conv4_3CP_num_within_layer_link_scores_pred(Featurelayer)
            within_layer_link_scores = tf.reshape(within_layer_link_scores,
                                                  within_layer_link_scores.get_shape().as_list()[:-1] + [8, 2])
            cross_layer_link_scores = None

        if layer_name == 'fc7':
            seg_scores = self.FC7_num_cls_pred(Featurelayer)
            seg_offsets = self.FC7_num_offset_pred(Featurelayer)
            within_layer_link_scores = self.FC7_num_within_layer_link_scores_pred(Featurelayer)
            within_layer_link_scores = tf.reshape(within_layer_link_scores,
                                                  within_layer_link_scores.get_shape().as_list()[:-1] + [8, 2])
            cross_layer_link_scores = self.FC7_num_cross_layer_link_scores_pred(Featurelayer)
            cross_layer_link_scores = tf.reshape(cross_layer_link_scores,
                                                 cross_layer_link_scores.get_shape().as_list()[:-1] + [4, 2])

        if layer_name == 'conv6_2':
            seg_scores = self.conv6_2CP_num_cls_pred(Featurelayer)
            seg_offsets = self.conv6_2CP_num_offset_pred(Featurelayer)
            within_layer_link_scores = self.conv6_2CP_num_within_layer_link_scores_pred(Featurelayer)
            within_layer_link_scores = tf.reshape(within_layer_link_scores,
                                                  within_layer_link_scores.get_shape().as_list()[:-1] + [8, 2])
            cross_layer_link_scores = self.conv6_2CP_num_cross_layer_link_scores_pred(Featurelayer)
            cross_layer_link_scores = tf.reshape(cross_layer_link_scores,
                                                 cross_layer_link_scores.get_shape().as_list()[:-1] + [4, 2])

        if layer_name == 'conv7_2':
            seg_scores = self.conv7_2CP_num_cls_pred(Featurelayer)
            seg_offsets = self.conv7_2CP_num_offset_pred(Featurelayer)
            within_layer_link_scores = self.conv7_2CP_num_within_layer_link_scores_pred(Featurelayer)
            within_layer_link_scores = tf.reshape(within_layer_link_scores,
                                                  within_layer_link_scores.get_shape().as_list()[:-1] + [8, 2])
            cross_layer_link_scores = self.conv7_2CP_num_cross_layer_link_scores_pred(Featurelayer)
            cross_layer_link_scores = tf.reshape(cross_layer_link_scores,
                                                 cross_layer_link_scores.get_shape().as_list()[:-1] + [4, 2])
        if layer_name == 'conv8_2':
            seg_scores = self.conv8_2CP_num_cls_pred(Featurelayer)
            seg_offsets = self.conv8_2CP_num_offset_pred(Featurelayer)
            within_layer_link_scores = self.conv8_2CP_num_within_layer_link_scores_pred(Featurelayer)
            within_layer_link_scores = tf.reshape(within_layer_link_scores,
                                                  within_layer_link_scores.get_shape().as_list()[:-1] + [8, 2])
            cross_layer_link_scores = self.conv8_2CP_num_cross_layer_link_scores_pred(Featurelayer)
            cross_layer_link_scores = tf.reshape(cross_layer_link_scores,
                                                 cross_layer_link_scores.get_shape().as_list()[:-1] + [4, 2])

        if layer_name == 'conv9_2':
            seg_scores = self.conv9_2CP_num_cls_pred(Featurelayer)
            seg_offsets = self.conv9_2CP_num_offset_pred(Featurelayer)
            within_layer_link_scores = self.conv9_2CP_num_within_layer_link_scores_pred(Featurelayer)
            within_layer_link_scores = tf.reshape(within_layer_link_scores,
                                                  within_layer_link_scores.get_shape().as_list()[:-1] + [8, 2])
            cross_layer_link_scores = self.conv9_2CP_num_cross_layer_link_scores_pred(Featurelayer)
            cross_layer_link_scores = tf.reshape(cross_layer_link_scores,
                                                 cross_layer_link_scores.get_shape().as_list()[:-1] + [4, 2])

        return seg_scores, seg_offsets, within_layer_link_scores, cross_layer_link_scores

    def build_loss(self, seg_labels, seg_offsets_labels, link_labels,seg_score_logits, seg_offsets, seg_scores, link_scores, link_score_logits):

        batch_size = seg_labels.shape[0]
        seg_neg_scores = seg_scores[:, :, 0]  # 得到segment是负样本的得分,其具体大小为(batch,N)

        #原文中作者到这里如果是Train with igonre则与下文无异，如果不带igonre训练，则将igonre完全除去，不算在neg or pos
        seg_pos_mask = tf.equal(seg_labels,1)
        seg_neg_mask = tf.equal(seg_labels,0)

        seg_selected_mask = self.OHNM_batch(seg_neg_scores, seg_pos_mask, seg_neg_mask, batch_size)
        n_seg_pos = tf.reduce_sum(tf.cast(seg_pos_mask, tf.float32))

        # 计算loss
        # do_summary = True
        link_cls_loss_weight = 1
        seg_loc_loss_weight = 5
        # ----------seg_cls_loss-----------
        with tf.name_scope('seg_cls_loss'):
            def has_pos():
                seg_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=seg_score_logits,
                    labels=tf.cast(seg_pos_mask, dtype=tf.int32))
                return tf.reduce_sum(seg_cls_loss * seg_selected_mask) / n_seg_pos

            def no_pos():
                return tf.constant(.0)

            seg_cls_loss = tf.cond(n_seg_pos > 0, has_pos, no_pos)
        # ----------seg_loc_loss-----------
        with tf.name_scope('seg_loc_loss'):
            def has_pos():
                # seg_pos_mask 很重要，只对pos的计算loss
                seg_loc_loss = self.smooth_l1_loss(seg_offsets, seg_offsets_labels,
                                              seg_pos_mask) * seg_loc_loss_weight / n_seg_pos

                return seg_loc_loss

            def no_pos():
                return tf.constant(.0)

            seg_loc_loss = tf.cond(n_seg_pos > 0, has_pos, no_pos)

        # -------------link_cls_loss---------------
        link_neg_scores = link_scores[:, :, 0]


        link_pos_mask = tf.equal(link_labels, 1)
        link_neg_mask = tf.equal(link_labels, 0)

        link_selected_mask = self.OHNM_batch(link_neg_scores, link_pos_mask, link_neg_mask, batch_size)
        n_link_pos = tf.reduce_sum(tf.cast(link_pos_mask, dtype=tf.float32))
        with tf.name_scope('link_cls_loss'):
            def has_pos():
                link_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=link_score_logits,
                    labels=tf.cast(link_pos_mask, tf.int32))
                return tf.reduce_sum(link_cls_loss * link_selected_mask) / n_link_pos

            def no_pos():
                return tf.constant(.0)

            link_cls_loss = tf.cond(n_link_pos > 0, has_pos, no_pos) * link_cls_loss_weight

        #loss = seg_cls_loss+seg_loc_loss+link_cls_loss
        #if loss == 0:
            #print("已保存")
            #pd.DataFrame(seg_labels).to_csv('{}_seg_label.csv'.format(index))
        return seg_cls_loss, seg_loc_loss, link_cls_loss

    def smooth_l1_loss(self, pred, target, weights):

        diff = pred - target
        abs_diff = tf.abs(diff)
        abs_diff_lt_1 = tf.less(abs_diff, 1)
        if len(target.shape) != len(weights.shape):
            loss = tf.reduce_sum(tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5), axis=2)
            return tf.reduce_sum(loss * tf.cast(weights, tf.float32))
        else:
            loss = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
        return tf.reduce_sum(loss * tf.cast(weights, tf.float32))

    def OHNM_single_image(self, scores, n_pos, neg_mask):
        """Online Hard Negative Mining.
            scores: the scores of being predicted as negative cls
            n_pos: the number of positive samples
            neg_mask: mask of negative samples
            Return:
                the mask of selected negative samples.
                if n_pos == 0, no negative samples will be selected.
        """

        def has_pos():
            n_neg = n_pos * 3  # 最多允许的负样本总数与正样本相比为1:3
            max_neg_entries = tf.reduce_sum(tf.cast(neg_mask, tf.int32))  # 这个batch里真正有的neg
            n_neg = tf.minimum(n_neg, max_neg_entries)
            n_neg = tf.cast(n_neg, tf.int32)
            neg_conf = tf.boolean_mask(scores, neg_mask)  # 那些真正是neg的被认为是neg的概率
            vals, _ = tf.nn.top_k(-neg_conf, k=n_neg)  # 把
            threshold = vals[-1]  # a negtive value,最小允许的值
            selected_neg_mask = tf.logical_and(neg_mask, scores <= -threshold)
            return tf.cast(selected_neg_mask, tf.float32)

        def no_pos():
            return tf.zeros_like(neg_mask, tf.float32)

        return tf.cond(n_pos > 0, has_pos, no_pos)

    def OHNM_batch(self, neg_conf, pos_mask, neg_mask, batch_size):
        selected_neg_mask = []
        for img_in_batch in range(batch_size):
            img_neg_poss = neg_conf[img_in_batch, :]
            img_neg_mask = neg_mask[img_in_batch, :]
            img_pos_mask = pos_mask[img_in_batch, :]
            n_pos = tf.reduce_sum(tf.cast(img_pos_mask, tf.int32))  # 计算一个img的bbox有多少正样本
            selected_neg_mask.append(self.OHNM_single_image(img_neg_poss, n_pos, img_neg_mask))
        selected_neg_mask = tf.stack(selected_neg_mask)  # 这是一个矩阵，行是一张img中对应的选取的neg样本
        selected_mask = tf.cast(pos_mask, tf.float32) + selected_neg_mask
        return selected_mask



