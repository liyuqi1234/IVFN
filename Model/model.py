import numpy as np
import tensorflow as tf
import math
import tensorflow.contrib.slim as slim
FLAGS = tf.app.flags.FLAGS


# tf.app.flags.DEFINE_float('lr','1e-4','learning rate')
def _variable_initializer(name, shape, initializer):
    # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    # dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_initializer(name, shape, tf.truncated_normal_initializer(stddev=stddev))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def conv2d(x, n_in, n_out, k, s, p='SAME', bias=False, scope='conv'):
    with tf.variable_scope(scope):
        kernel = _variable_with_weight_decay('weights',
                                             shape=[k, k, n_in, n_out],
                                             stddev=np.sqrt(2 / (k * k * n_in)),
                                             wd=None)
        # kernel=tf.get_variable('weights',tf.truncated_normal([k,k,n_in,n_out],stddev=np.sqrt(2/(k*k*n_in))))
        # tf.add_to_collection('weights',kernel)
        conv = tf.nn.conv2d(x, kernel, [1, s, s, 1], padding=p)
        if bias:
            bias = _variable_initializer('bias', [n_out], initializer=tf.constant_initializer(0.0))
            # tf.add_to_collection('biases',bias)
            conv = tf.nn.bias_add(conv, bias)
        return conv


def _batch_norm(x, is_training, name=None):
    return tf.contrib.layers.batch_norm(inputs=x,
                                        decay=0.95,
                                        center=True,
                                        scale=True,
                                        is_training=is_training,
                                        updates_collections=None,
                                        scope=(name + '_batch_norm'))


def blockLayer(x, channels, r, kernel_size):
    output = tf.layers.conv2d(x, channels, kernel_size, padding='same', dilation_rate=(r, r), use_bias=False)
    return tf.nn.relu(output)
def resDenseBlock(x, channels, layers=3, kernel_size=[3,3], scale=1):
    outputs = [x]
    rates = [1]*layers
    for i in range(layers):
        output = blockLayer(tf.concat(outputs[:i],3) if i>=1 else x, channels, rates[i],kernel_size)
        outputs.append(output)

    output = tf.concat(outputs, 3)
    output = slim.conv2d(output, channels, [1,1])
    output *= scale
    return x + output

def upsample(x, scale=16, features=32):
    output = x
    if (scale & (scale-1)) == 0:
        for _ in range(int(math.log(scale, 2))):
            output = tf.layers.conv2d(output, 4*features, (3, 3), padding='same', use_bias=False)
            output = pixelshuffle(output, 2)
    elif scale == 3:
        output = tf.layers.conv2d(output, 9*features, (3, 3), padding='same', use_bias=False)
        output = pixelshuffle(output, 3)
    else:
        raise NotImplementedError
    return output

def pixelshuffle(x, upscale_factor):
    return tf.depth_to_space(x, upscale_factor)

def condition(x,is_training):
    result = tf.where(is_training,tf.nn.dropout(x , keep_prob = 0.8), x)
    return result 

def inference(images,infrareds,global_layers,local_layers,scaling_factor,is_training):
    with tf.variable_scope('RDB'):
        ##__________________________________________________________________________________________________________scale1
        with tf.variable_scope('conv1_scale1'):
            # _input = images[0:1, :, :, 0:4]
            # _input = tf.transpose(_input, perm=[3,1,2,0])
            #tf.summary.image('visible_input', _input)#——————————————可视化输入
            conv1_VI = conv2d(images, 3, 96, 11, 4, p='SAME', bias=True, scope='conv1_VI')#64*128*96
            conv1_VI_relu = tf.nn.relu(_batch_norm(conv1_VI, is_training, name='bn1_VI'))
            # image = conv1_VI_relu[0:1, :, :, 0:16]
            # image = tf.transpose(image, perm=[3,1,2,0])
            #tf.summary.image('Image_output_conv1', image)#————————————————可视化第一层卷积
            pool1_VI = tf.nn.max_pool(conv1_VI_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1_VI')#32*64*96

            conv1_IR = conv2d(infrareds, 3, 96, 11, 4, p='SAME', bias=True, scope='conv1_IR')#64*128*96
            conv1_IR_relu = tf.nn.relu(_batch_norm(conv1_IR, is_training, name='bn1_IR'))
            pool1_IR = tf.nn.max_pool(conv1_IR_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool1_IR')#32*64*96
        with tf.variable_scope('conv2_scale1'):#耦合率为0.25
            kernel2_VI = _variable_with_weight_decay('weights2_VI',
                                                        shape=[5, 5, 96, 192],
                                                        stddev=np.sqrt(2 / (5 * 5 * 96)),
                                                        wd=None)
            kernel2_IR = _variable_with_weight_decay('weights2_IR',
                                                        shape=[5, 5, 96, 192],
                                                        stddev=np.sqrt(2 / (5 * 5 * 96)),
                                                        wd=None)
            kernel2_share = _variable_with_weight_decay('weights2_share',
                                                        shape=[5, 5, 96, 64],
                                                        stddev=np.sqrt(2 / (5 * 5 * 96)),
                                                        wd=None)
            bias2_VI = _variable_initializer('bias2_VI', [256], initializer=tf.constant_initializer(0.0))
            conv2_VI = tf.nn.conv2d(pool1_VI, tf.concat([kernel2_VI,kernel2_share],3), [1, 1, 1, 1], padding='SAME')
            conv2_VI_relu = tf.nn.relu(_batch_norm(conv2_VI, is_training, name='bn2_VI'))
            pool2_VI = tf.nn.max_pool(conv2_VI_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2_VI')#16*32*256

            bias2_IR = _variable_initializer('bias2_IR', [256], initializer=tf.constant_initializer(0.0))
            conv2_IR = tf.nn.conv2d(pool1_IR, tf.concat([kernel2_IR,kernel2_share],3), [1, 1, 1, 1], padding='SAME')
            conv2_IR_relu = tf.nn.relu(_batch_norm(conv2_IR, is_training, name='bn2_IR'))
            pool2_IR = tf.nn.max_pool(conv2_IR_relu,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME',name='pool2_IR')#16*32*256

        with tf.variable_scope('conv3_scale1'):#耦合率为0.5
            kernel3_VI = _variable_with_weight_decay('weights3_VI',
                                                        shape=[3, 3, 256, 192],
                                                        stddev=np.sqrt(2 / (3 * 3 * 256)),
                                                        wd=None)
            kernel3_IR = _variable_with_weight_decay('weights3_IR',
                                                        shape=[3, 3, 256, 192],
                                                        stddev=np.sqrt(2 / (3 * 3 * 256)),
                                                        wd=None)
            kernel3_share = _variable_with_weight_decay('weights3_share',
                                                        shape=[3, 3, 256, 192],
                                                        stddev=np.sqrt(2 / (3 * 3 * 256)),
                                                        wd=None)
            bias3_VI = _variable_initializer('bias3_VI', [384], initializer=tf.constant_initializer(0.0))
            conv3_VI = tf.nn.conv2d(pool2_VI, tf.concat([kernel3_VI,kernel3_share],3), [1, 1, 1, 1], padding='SAME')
            conv3_VI_relu = tf.nn.relu(_batch_norm(conv3_VI, is_training, name='bn3_VI'))#16*32*384
            conv3_VI_relu = condition(conv3_VI_relu ,is_training)

            bias3_IR = _variable_initializer('bias3_IR', [384], initializer=tf.constant_initializer(0.0))
            conv3_IR = tf.nn.conv2d(pool2_IR, tf.concat([kernel3_IR,kernel3_share],3), [1, 1, 1, 1], padding='SAME')
            conv3_IR_relu = tf.nn.relu(_batch_norm(conv3_IR, is_training, name='bn3_IR'))#16*32*384
            conv3_IR_relu = condition(conv3_IR_relu ,is_training)
        with tf.variable_scope('conv4_scale1'):#耦合率为0.75
            kernel4_VI = _variable_with_weight_decay('weights4_VI',
                                                        shape=[3, 3, 384, 96],
                                                        stddev=np.sqrt(2 / (3 * 3 * 384)),
                                                        wd=None)
            kernel4_IR = _variable_with_weight_decay('weights4_IR',
                                                        shape=[3, 3, 384, 96],
                                                        stddev=np.sqrt(2 / (3 * 3 * 384)),
                                                        wd=None)
            kernel4_share = _variable_with_weight_decay('weights4_share',
                                                        shape=[3, 3, 384, 288],
                                                        stddev=np.sqrt(2 / (3 * 3 * 384)),
                                                        wd=None)
            bias4_VI = _variable_initializer('bias4_VI', [384], initializer=tf.constant_initializer(0.0))
            conv4_VI = tf.nn.conv2d(conv3_VI_relu, tf.concat([kernel4_VI,kernel4_share],3), [1, 1, 1, 1], padding='SAME')
            conv4_VI_relu = tf.nn.relu(_batch_norm(conv4_VI, is_training, name='bn4_VI'))#16*32*384
            conv4_VI_relu = condition(conv4_VI_relu ,is_training)

            bias4_IR = _variable_initializer('bias4_IR', [384], initializer=tf.constant_initializer(0.0))
            conv4_IR = tf.nn.conv2d(conv3_IR_relu, tf.concat([kernel4_IR,kernel4_share],3), [1, 1, 1, 1], padding='SAME')
            conv4_IR_relu = tf.nn.relu(_batch_norm(conv4_IR, is_training, name='bn4_IR'))#16*32*384
            conv4_IR_relu = condition(conv4_IR_relu ,is_training)
        with tf.name_scope('fc6') as scope:
            fc6_VI = conv2d(conv4_VI_relu, 384, 32, 1, 1, p='SAME', bias=True, scope='fc6_VI')
            image_VI = fc6_VI[0:1, :, :, 0:16]
            image_VI = tf.transpose(image_VI, perm=[3,1,2,0])
            tf.summary.image('VI_output_fc6', image_VI)
            fc6_IR = conv2d(conv4_IR_relu, 384, 32, 1, 1, p='SAME', bias=True, scope='fc6_IR')#16*32*32
            image_IR = fc6_IR[0:1, :, :, 0:16]
            image_IR = tf.transpose(image_IR, perm=[3,1,2,0])
            tf.summary.image('IR_output_fc6', image_IR)
        with tf.name_scope('deconv_image') as scope:
            wt4_VI=tf.Variable(tf.truncated_normal([3,3,256,384]))
            deconv4_image=tf.nn.relu(tf.nn.conv2d_transpose(conv4_VI_relu, wt4_VI, [FLAGS.batch_size,32,64,256], [1,2,2,1], padding='SAME'))
            wt5_VI=tf.Variable(tf.truncated_normal([3,3,256,256]))
            deconv5_image=tf.nn.relu(tf.nn.conv2d_transpose(deconv4_image, wt5_VI, [FLAGS.batch_size,64,128,256], [1,2,2,1], padding='SAME'))#64*128*256
        with tf.name_scope('deconv_infrared') as scope:
            wt4_IR=tf.Variable(tf.truncated_normal([3,3,256,384]))
            deconv4_infrared=tf.nn.relu(tf.nn.conv2d_transpose(conv4_IR_relu, wt4_IR, [FLAGS.batch_size,32,64,256], [1,2,2,1], padding='SAME'))
            wt5_IR=tf.Variable(tf.truncated_normal([3,3,256,256]))
            deconv5_infrared=tf.nn.relu(tf.nn.conv2d_transpose(deconv4_infrared, wt5_IR, [FLAGS.batch_size,64,128,256], [1,2,2,1], padding='SAME'))#64*128*256
        with tf.variable_scope('conv6_scale1'):
            fc6_VI = conv2d(deconv5_image, 256, 32, 1, 1, p='SAME', bias=True, scope='fc6_VI')#64*128*32
            # image_VI = fc6_VI[0:1, :, :, 0:16]
            # image_VI = tf.transpose(image_VI, perm=[3,1,2,0])
            # tf.summary.image('VI_output_fc6', image_VI)
            fc6_VI = condition(fc6_VI,is_training)
            fc6_IR = conv2d(deconv5_infrared, 256, 32, 1, 1, p='SAME', bias=True, scope='fc6_IR')
            # image_IR = fc6_VI[0:1, :, :, 0:16]
            # image_IR = tf.transpose(image_IR, perm=[3,1,2,0])
            # tf.summary.image('IR_output_fc6', image_IR)
            fc6_IR = condition(fc6_IR,is_training)
        #_______________________________________________________________________________________________将红外和可见光特征融合
        fea_concat = tf.concat([fc6_VI,fc6_IR],3)#16*32*64
        with tf.variable_scope('IR_VI_fuse'):
            concat_IR_VI = conv2d(fea_concat, 64, 32, 1, 1, p='SAME', bias=True, scope='concat_IR_VI')#16*32*32
            concat_IR_VI_relu = tf.nn.relu(_batch_norm(concat_IR_VI, is_training, name='bn_concat'))
        coe_matrix_rgb = tf.nn.sigmoid(concat_IR_VI_relu)
        coe_matrix_ir = tf.ones(
            (fc6_VI.shape[0], fc6_VI.shape[1], fc6_VI.shape[2], fc6_VI.shape[3])) - coe_matrix_rgb
        prob_ir = fc6_IR * coe_matrix_ir
        # image_IR_prob = prob_ir[0:1, :, :, 0:16]
        # image_IR_prob = tf.transpose(image_IR_prob, perm=[3,1,2,0])
        # tf.summary.image('image_IR_prob', image_IR_prob)
        prob_rgb = fc6_VI * coe_matrix_rgb
        # image_VI_prob = prob_rgb[0:1, :, :, 0:16]
        # image_VI_prob = tf.transpose(image_VI_prob, perm=[3,1,2,0])
        # tf.summary.image('image_VI_prob', image_VI_prob)
        fusion = tf.add(prob_ir, prob_rgb)#16*32*32
        # image_fusion = fusion[0:1, :, :, 0:16]
        # image_fusion = tf.transpose(image_fusion, perm=[3,1,2,0])
        # tf.summary.image('image_fusion', image_fusion)
        # ##________________________________________________________________________________________________RDB
        with tf.variable_scope('RDN_conv'):
            F_1 = conv2d(fusion, 32, 32, 3, 1, p='SAME', bias=False,scope="conv_F_1")
            F_1 = tf.nn.relu(_batch_norm(F_1, is_training, name='bn_conv_F_1'))#16*32*32
            F_2 = conv2d(F_1, 32, 32, 3, 1, p='SAME', bias=False,scope="conv_F_2")
            F_2 = tf.nn.relu(_batch_norm(F_2, is_training, name='bn_conv_F_2'))#16*32*32
            F_2 = condition (F_2,is_training)
        with tf.variable_scope('RDN_IR_VI'):
            outputs = []
            for i in range(global_layers):
                F_2 = resDenseBlock(F_2, channels=32, layers=local_layers, kernel_size=[3,3],scale=scaling_factor)
                outputs.append(F_2)
            F_D = tf.concat(outputs, 3)#16*32*96
            FGF1 = conv2d(F_D, 96, 32, 1, 1, p='SAME', bias=False, scope='conv_FGF1')#16*32*32
            FGF2 = conv2d(FGF1, 32, 32, 3, 1, p='SAME', bias=False, scope='conv_FGF2')#16*32*32
            FGF2 = tf.nn.relu(_batch_norm(FGF2, is_training, name='conv_FGF2'))
            FDF = tf.add(FGF2, F_1)#16*32*32
            deconv = upsample(fusion)
        output = tf.layers.conv2d(deconv, 32, (1, 1), padding='same', use_bias=False)
        return output

def acc(pred,lables):
    pred = tf.argmax(pred,axis=3)
    pred = tf.expand_dims(pred, -1)
    #labels = tf.reshape(lables, [-1])
    #pred = tf.reshape(pred, [-1])
    tf.cast(pred,tf.int64)
    correct_predictions = tf.to_float(
                tf.equal(pred, lables))
    acc = tf.reduce_mean(correct_predictions)
    return acc

def loss(pre_dep, true_dep):
    dim = pre_dep.get_shape()[3].value
    logits = tf.reshape(pre_dep, [-1, dim])
    labels = tf.reshape(true_dep, [-1])
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                        name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)
    weight l2 decay loss
    weight_l2_losses = [tf.nn.l2_loss(o) for o in tf.get_collection('weights')]
    weight_decay_loss = tf.multiply(1e-4, tf.add_n(weight_l2_losses),
                                   name='weight_decay_loss')
    tf.add_to_collection('losses', weight_decay_loss)
    total loss
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train(loss, global_step):
    learning_rate = tf.train.exponential_decay(1e-4, global_step, 20000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op




        
        







            








