import abc
from enum import Enum
import os
import tensorflow as tf
from .flowlib import flow_to_image, write_flow
import numpy as np
from scipy.misc import imread, imsave, imresize
import uuid
from .training_schedules import LONG_SCHEDULE
import cv2
slim = tf.contrib.slim


class Mode(Enum):
    TRAIN = 1
    TEST = 2


class Net(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, mode=Mode.TRAIN, debug=False):
        self.global_step = slim.get_or_create_global_step()
        self.mode = mode
        self.debug = debug

    @abc.abstractmethod
    def model(self, inputs, training_schedule, trainable=True):
        """
        Defines the model and returns a tuple of Tensors needed for calculating the loss.
        """
        return

    @abc.abstractmethod
    def loss(self, **kwargs):
        """
        Accepts prediction Tensors from the output of `model`.
        Returns a single Tensor representing the total loss of the model.
        """
        return

    def test(self, checkpoint, input_a_path, input_b_path, out_path, save_image=True, save_flo=True):
        input_a = imread(input_a_path)
        input_b = imread(input_b_path)

        # Convert from RGB -> BGR
        input_a = input_a[..., [2, 1, 0]]
        input_b = input_b[..., [2, 1, 0]]

        # Scale from [0, 255] -> [0.0, 1.0] if needed
        if input_a.max() > 1.0:
            input_a = input_a / 255.0
        if input_b.max() > 1.0:
            input_b = input_b / 255.0

        # TODO: This is a hack, we should get rid of this
        training_schedule = LONG_SCHEDULE

        inputs = {
            'input_a': tf.expand_dims(tf.constant(input_a, dtype=tf.float32), 0),
            'input_b': tf.expand_dims(tf.constant(input_b, dtype=tf.float32), 0),
        }
        predictions = self.model(inputs, training_schedule)
        pred_flow = predictions['flow']

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            pred_flow = sess.run(pred_flow)[0, :, :, :]

            unique_name = 'flow-' + str(uuid.uuid4())
            if save_image:
                flow_img = flow_to_image(pred_flow)
                full_out_path = os.path.join(out_path, unique_name + '.png')
                imsave(full_out_path, flow_img)

            if save_flo:
                full_out_path = os.path.join(out_path, unique_name + '.flo')
                write_flow(pred_flow, full_out_path)

    def train(self, log_dir, training_schedule, input_a, input_b, flow, checkpoints=None):
        tf.summary.image("image_a", input_a, max_outputs=2)
        tf.summary.image("image_b", input_b, max_outputs=2)

        self.learning_rate = tf.train.piecewise_constant(
            self.global_step,
            [tf.cast(v, tf.int64) for v in training_schedule['step_values']],
            training_schedule['learning_rates'])

        optimizer = tf.train.AdamOptimizer(
            self.learning_rate,
            training_schedule['momentum'],
            training_schedule['momentum2'])

        inputs = {
            'input_a': input_a,
            'input_b': input_b,
        }
        predictions = self.model(inputs, training_schedule)
        total_loss = self.loss(flow, predictions)
        tf.summary.scalar('loss', total_loss)

        if checkpoints:
            for (checkpoint_path, (scope, new_scope)) in checkpoints.iteritems():
                variables_to_restore = slim.get_variables(scope=scope)
                renamed_variables = {
                    var.op.name.split(new_scope + '/')[1]: var
                    for var in variables_to_restore
                }
                restorer = tf.train.Saver(renamed_variables)
                with tf.Session() as sess:
                    restorer.restore(sess, checkpoint_path)

        # Show the generated flow in TensorBoard
        if 'flow' in predictions:
            pred_flow_0 = predictions['flow'][0, :, :, :]
            pred_flow_0 = tf.py_func(flow_to_image, [pred_flow_0], tf.uint8)
            pred_flow_1 = predictions['flow'][1, :, :, :]
            pred_flow_1 = tf.py_func(flow_to_image, [pred_flow_1], tf.uint8)
            pred_flow_img = tf.stack([pred_flow_0, pred_flow_1], 0)
            tf.summary.image('pred_flow', pred_flow_img, max_outputs=2)

        true_flow_0 = flow[0, :, :, :]
        true_flow_0 = tf.py_func(flow_to_image, [true_flow_0], tf.uint8)
        true_flow_1 = flow[1, :, :, :]
        true_flow_1 = tf.py_func(flow_to_image, [true_flow_1], tf.uint8)
        true_flow_img = tf.stack([true_flow_0, true_flow_1], 0)
        tf.summary.image('true_flow', true_flow_img, max_outputs=2)

        train_op = slim.learning.create_train_op(
            total_loss,
            optimizer,
            summarize_gradients=True)

        if self.debug:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tf.train.start_queue_runners(sess)
                slim.learning.train_step(
                    sess,
                    train_op,
                    self.global_step,
                    {
                        'should_trace': tf.constant(1),
                        'should_log': tf.constant(1),
                        'logdir': log_dir + '/debug',
                    }
                )
        else:
            slim.learning.train(
                train_op,
                log_dir,
                # session_config=tf.ConfigProto(allow_soft_placement=True),
                global_step=self.global_step,
                save_summaries_secs=60,
                number_of_steps=training_schedule['max_iter']
            )



    def test_kitti(self, checkpoint, input_a_path, input_b_path, out_path, save_image=True, save_flo=True):


        input_as = []
        input_bs = []

        height_o = []
        width_o = []

        for i in range(len(input_a_path)):

            a = imread(input_a_path[i])
            b = imread(input_b_path[i])

            a_height = a.shape[0]
            a_width = a.shape[1]

            height_o.append(a_height)
            width_o.append(a_width)

            input_a = imresize(a, (384, 512))
            input_b = imresize(b, (384, 512))

            # Convert from RGB -> BGR
            input_a = input_a[..., [2, 1, 0]]
            input_b = input_b[..., [2, 1, 0]]


            # Scale from [0, 255] -> [0.0, 1.0] if needed
            if input_a.max() > 1.0:
                input_a = input_a / 255.0
            if input_b.max() > 1.0:
                input_b = input_b / 255.0

            input_as.append(input_a)
            input_bs.append(input_b)

        print('net: number of samples: {}'.format(len(input_as)))


        # TODO: This is a hack, we should get rid of this
        training_schedule = LONG_SCHEDULE

        inputs = {
            'input_a': tf.placeholder(shape=[1, 384, 512, 3], dtype=tf.float32),
            'input_b': tf.placeholder(shape=[1, 384, 512, 3], dtype=tf.float32),
        }

        predictions = self.model(inputs, training_schedule)
        pred_flow = predictions['flow']

        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, checkpoint)


        for i in range(len(input_as)):

            # inputs_constant = {
            #     'input_a': tf.expand_dims(tf.constant(input_as[i], dtype=tf.float32), 0),
            #     'input_b': tf.expand_dims(tf.constant(input_bs[i], dtype=tf.float32), 0),
            # }

            pred_flow_out = sess.run(pred_flow, feed_dict={inputs['input_a']: input_as[i][np.newaxis, ...],
                                                       inputs['input_b']: input_bs[i][np.newaxis, ...]})[0, :, :, :]

            # pred_flow = sess.run(pred_flow, feed_dict={inputs: inputs_constant})[0, :, :, :]

            print('processing {}/{} ...'.format(i + 1, len(input_as)))

            unique_name = 'flow-' + str(i).zfill(6) + '_10'
            if save_image:
                flow_img = flow_to_image(pred_flow_out)

                flow_img = imresize(flow_img, (height_o[i], width_o[i]))

                full_out_path = os.path.join(out_path, unique_name + '.png')

                imsave(full_out_path, flow_img)

            if save_flo:

                u = cv2.resize(pred_flow_out[:, :, 0], (width_o[i], height_o[i]))
                v = cv2.resize(pred_flow_out[:, :, 1], (width_o[i], height_o[i]))

                u = u.reshape(height_o[i], width_o[i], 1)
                v = v.reshape(height_o[i], width_o[i], 1)

                pred_flow_resize = np.concatenate((u, v), axis=2)


                full_out_path = os.path.join(out_path, unique_name + '.flo')
                write_flow(pred_flow_resize, full_out_path)

        print('done')


