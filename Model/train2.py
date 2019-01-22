
import tensorflow as tf
from datetime import datetime
import model_EIGEN as m
import input_net
import h5py
import time

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('max_step', 80000, 'Number of max step')
tf.app.flags.DEFINE_string('batch_size', 4, 'batch size')
#tf.app.flags.DEFINE_string('log_dir', '../logs_net/log_%s' % time.strftime("%Y%m%d_%H%M%S"), '')
#tf.app.flags.DEFINE_integer('summary_interval', 1, 'Interval for summary.')



def write_hdf5(file_name, losses):
    with h5py.File(file_name, 'w') as f:
        f.create_dataset('losses', data=losses)

def train():
    data =input_net.read_file('data1.h5')
    with tf.Graph().as_default():
        images = tf.placeholder(tf.float32, [FLAGS.batch_size, 256, 512, 3])
        infrareds = tf.placeholder(tf.float32, [FLAGS.batch_size, 256,512,3])
        labels = tf.placeholder(tf.int64, [FLAGS.batch_size, 256,512, 1])
        global_step = tf.Variable(1, trainable=False)
        is_training = tf.placeholder(tf.bool)

        pre_dep = m.inference(infrareds, images, global_layers=3, local_layers=3, scaling_factor=1, is_training=True)
        loss = m.loss(pre_dep, labels)

        train_op = m.train(loss, global_step)
        saver = tf.train.Saver()
        losses_save = []


        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('C:/logs_layer/',graph = sess.graph)
            sess.run(tf.initialize_all_variables())
            format_str = "%s    Iteration =  %d   loss = %0.3f"
            for step in range(FLAGS.max_step + 1):

                batch = data.next_batch(FLAGS.batch_size)
                loss_val, summary_str, _ = sess.run([loss, merged, train_op], feed_dict=
                            {images: batch[0], infrareds: batch[1], labels: batch[2],is_training:True})
    
                
                if step % 20 == 0:
                    print(format_str%(datetime.now(), step, loss_val))
                    losses_save.append(loss_val)
  
                if step % 10000 == 0:
                    checkpoint = 'saver/' + 'model.ckpt'
                    saver.save(sess, checkpoint, global_step=step) 

                if step % 80000 == 0:
                    write_hdf5('vgg_loss.h5', losses_save)
                    train_writer.add_summary(summary_str, step)


def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()

