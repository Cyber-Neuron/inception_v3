# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import os.path
import time


import numpy as np
import tensorflow as tf

from inception import image_processing
from inception import inception_model as inception


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval',
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train',
                           """Directory where to read model checkpoints.""")

# Flags governing the frequency of the eval.
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

# Flags governing the data used for the eval.
tf.app.flags.DEFINE_integer('num_examples', 50000,
                            """Number of examples to run. Note that the eval """
                            """ImageNet dataset contains 50000 examples.""")
tf.app.flags.DEFINE_string('subset', 'test',
                           """Either 'test'.""")


def _eval_once(saver, top_1_op, filenames,labels):
  """Runs Eval once.

  Args:
    saver: Saver.
    top_1_op: Top 1 op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      if os.path.isabs(ckpt.model_checkpoint_path):
        # Restores from checkpoint with absolute path.
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        # Restores from checkpoint with relative path.
        saver.restore(sess, os.path.join(FLAGS.checkpoint_dir,
                                         ckpt.model_checkpoint_path))

      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/imagenet_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      print('Succesfully loaded model from %s at step=%s.' %
            (ckpt.model_checkpoint_path, global_step))
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))

      num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
      #Only run once
      num_iter = 1 
      # Counts the number of correct predictions.
      count_top_1 = 0.0
      total_sample_count = num_iter * FLAGS.batch_size
      step = 0

      print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
      start_time = time.time()
      fnames = filenames.eval()
      labe = labels.eval()
      while step < num_iter and not coord.should_stop():
        top_1 = sess.run([top_1_op])
        indx=0
        for top in top_1[0]:
         print('%s %s' % (fnames[indx],top))
         indx+=1
        break
        step += 1
      # Compute precision @ 1.

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(dataset):
  """Evaluate model on Dataset for a number of steps."""
  with tf.Graph().as_default():
    # Get images and labels from the dataset.
    images, labels, filenames = image_processing.inputs(dataset,batch_size=FLAGS.batch_size)
    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    num_classes = dataset.num_classes() + 1

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, _ = inception.inference(images, num_classes)
    # Calculate predictions.
    #top_1_op = tf.nn.in_top_k(logits, labels, 1)
    top_1_op = tf.argmax(logits, 1)
    #top_1_op = logits
    #top_1_op = tf.nn.softmax(logits)
    
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        inception.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)


    graph_def = tf.get_default_graph().as_graph_def()

    while True:
      _eval_once(saver, top_1_op, filenames,labels)
      break
