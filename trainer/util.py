import numpy as np
import tensorflow as tf
import tensornets as nets

CSV_COLUMNS=['emotion','pixels']
CSV_COLUMN_DEFAULTS=[[0], ['']]

def parse_csv(inputs):
    row_columns=tf.expand_dims(inputs, -1)
    columns=tf.decode_csv(row_columns, 
                            record_defaults=CSV_COLUMN_DEFAULTS, 
                            field_delim=',')
    features=dict(zip(CSV_COLUMNS, columns))
    
    y=features['emotion']
    y=tf.one_hot(y, depth = 7) 
    y=tf.reshape(y, [7])

    x=features['pixels']
    x=tf.string_split(x) 
    x=tf.sparse_tensor_to_dense(x, default_value='0')
    x=tf.string_to_number(x)
    x=tf.reshape(x, [48,48,1])
    x=tf.image.grayscale_to_rgb(x)
    x=tf.image.resize_images(x, [224,224])

    return x, y