import trainer.util as util
import tensorflow as tf
import tensornets as nets
from tensorflow.contrib.training.python.training import hparam

import argparse

def run_model(hparams):
    train_dataset=tf.data.TextLineDataset(hparams.train_file) \
        .skip(1) \
        .map(util.parse_csv) \
        .shuffle(30000) \
        .repeat() \
        .batch(hparams.train_batch_size) \
        .prefetch(1)
    
    valid_dataset=tf.data.TextLineDataset(hparams.eval_file) \
        .skip(1) \
        .map(util.parse_csv) \
        .repeat() \
        .batch(hparams.eval_batch_size) \
        .prefetch(1)

    iterator=tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    x, y=iterator.get_next()

    training_init_op=iterator.make_initializer(train_dataset)
    validation_init_op=iterator.make_initializer(valid_dataset)

    is_train=tf.placeholder_with_default(False, shape=(), name="is_train")
    model=nets.DenseNet169(x, is_training=is_train, classes=7)
    train_list=model.get_weights()[hparams.first_layer:]  # 520 = only retrain last conv block
    loss=tf.losses.softmax_cross_entropy(y, model)

    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train=tf.train.AdamOptimizer(learning_rate=hparams.lr).minimize(loss,var_list=train_list) # minimize(loss, var_list=train_list)
    
    test_acc, test_acc_op=tf.metrics.accuracy(tf.argmax(y, 1),tf.argmax(model,1))

    init_op=tf.global_variables_initializer()
    local_init_op=tf.local_variables_initializer()

    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        sess.run(local_init_op)
        sess.run(training_init_op)
        sess.run(model.pretrained())

        saver=tf.train.Saver()
        if hparams.load_ckpt==1:
            saver.restore(sess, hparams.ckpt_in + ".ckpt")

        for epoch in range(hparams.num_epochs):
             for _ in range(int(28709/hparams.train_batch_size)):
                 sess.run(train, {is_train: True})
             l=sess.run(loss, {is_train: True})
             print("Epoch: {}, loss: {:.3f}".format(epoch, l))
             
            #  reinitialize iterator  with validation data
             sess.run(validation_init_op)
             eval_iters=int(7178/hparams.train_batch_size)
             for _ in range(eval_iters):
                 sess.run(test_acc_op, {is_train: False})

             print("Accumulated validation set accuracy over {} iterations is {:.2f}%".format(eval_iters,
             sess.run(test_acc)*100))
             save_path=saver.save(sess, hparams.ckpt_out + "{}.ckpt".format(epoch))
             print("Model saved in path: %s" % save_path)

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      "--train-file",
      help="GCS or local paths to training data",
      required=True
  )
  parser.add_argument(
      "--num-epochs",
      help="Maximum number of training data epochs on which to train",
      type=int
  )
  parser.add_argument(
      "--lr",
      help="Learning rate",
      type=float,
      default=1e-5
  )
  parser.add_argument(
      "--train-batch-size",
      help="Batch size for training steps",
      type=int,
      default=40
  )
  parser.add_argument(
      "--eval-batch-size",
      help="Batch size for evaluation steps",
      type=int,
      default=40
  )
  parser.add_argument(
      "--eval-file",
      help="GCS or local paths to evaluation data",
      required=True
  )
  parser.add_argument(
      "--ckpt-in",
      help="Name of the input ckpt file without the suffix",
  )
  parser.add_argument(
      "--ckpt-out",
      help="Name of the output ckpt file without the suffix",
      required=True
  )
  parser.add_argument(
      "--load-ckpt",
      help="Whether to restore a checkpoint",
      type=int,
      default=0,
  )
  parser.add_argument(
      "--first-layer",
      help="First layer to train from",
      type=int,
      default=0,
  )

  args = parser.parse_args()

  # Run the training job
  hparams=hparam.HParams(**args.__dict__)
  run_model(hparams)