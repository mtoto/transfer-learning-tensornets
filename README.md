# Transfer Learning with Tensornets and the Dataset API

This repo contains the complete code from the [blog post Transfer Learning with Tensornets and Dataset API](http://tamaszilagyi.com/blog/2019/2019-01-12-tensornets/). The goal of this project is to tackle the the [2013 Facial Expression Recognition Challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) using transfer learning implemented with Tensorflow's high-level API's. 

# Data

You can download the data from the above link, copy it to the `data/` folder and subsequently run `./datagen.sh` in your Terminal. This function divides the `fer2013.csv` file into `train` and `validation` sets.

# Training the model on Google Cloud

While it is possible to train this model locally, I did so on [Goolge's ML Engine](https://cloud.google.com/ml-engine/) with the `gcloud` command-line tool. Make sure you uplaod both .csv files to a Google Cloud Storage bucket first. The trainer program and its utility functions are located in the `./train/` folder. An example script is (to be run from the root of this directory) would look something like the following.

```bash
gcloud ml-engine jobs submit training 'jobname' --region europe-west1 \
                           --scale-tier 'basic-gpu' \
                           --package-path trainer/ \
                           --staging-bucket 'gs://bucketname' \
                           --runtime-version 1.9 \
                           --module-name trainer.task \
                           --python-version 3.5 \
                           --packages deps/Cython-0.29.2.tar.gz,deps/tensornets-0.3.6.tar.gz \
                           -- \
                           --train-file 'gs://path/to/train.csv' \
                           --eval-file 'gs://path/to/valid.csv' \
                           --num-epochs 8 \
                           --train-batch-size 32 \
                           --eval-batch-size 32 \
                           --ckpt-out 'gs://path/to/save/model'
```

# Inferece

When the job finishes, the checkpoint files are saved to the location you specified (`'gs://path/to/save/model'` above). After downloading the files, we need to freeze the weights and optimize the graph for inference. This can be done using the `trainer/freeze.py`. From the command-line:

```bash
./freeze.py --ckpt ~/path/to/model.ckpt --out-file frozen_model.pb
```

Now we can use the `.pb` file for inference, for example testing the model on our webcam:

```bash
./videocam.py --cascade-file face_default.xml --pb-file path/to/frozen_model.pb
```

![](https://raw.githubusercontent.com/mtoto/mtoto.github.io/master/blog/2019/emotins.png) 
