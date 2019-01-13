#!/usr/bin/env python3
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
import tensorflow.tools.graph_transforms as graph_transforms
from tensorflow.python.platform import gfile
import tensornets as nets

def freeze_and_optimize(ckpt, out_file):
    iterator = tf.data.Iterator.from_structure((tf.float32, tf.float32),((None, 224, 224, 3), (None,7)))
    x, _ = iterator.get_next()    
    model = nets.DenseNet169(x, is_training=False, classes=7)
    init_op = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        # freeze         
        frozen_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, 
            tf.get_default_graph().as_graph_def(),
            ["densenet169/probs"])
        # transform
        tr_graph_def=graph_transforms.TransformGraph(frozen_graph_def,
                              inputs=["IteratorGetNext"],
                              outputs=["densenet169/probs"],
                              transforms=["strip_unused_nodes",
                              "fold_constants", "fold_batch_norms", 
                              "fold_old_batch_norms", "sort_by_execution_order"])
        # optimize
        opt_graph_def = optimize_for_inference_lib.optimize_for_inference(
            tr_graph_def, ["IteratorGetNext"], ["densenet169/probs"],
        tf.float32.as_datatype_enum)
        
        with tf.gfile.GFile(out_file, "wb") as f:
            f.write(opt_graph_def.SerializeToString())
        
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--ckpt',
      help='Path to the checkpoint files.',
      type=str,
      required=True
  )
  parser.add_argument(
      '--out-file',
      help='Output file name',
      type=str,
      required=True
  )
  args = parser.parse_args()
  # Freeeeze!
  freeze_and_optimize(**args.__dict__)