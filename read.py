

import tensorflow as tf

for summary in tf.compat.v1.train.summary_iterator("exp_out/nq/t5_1.1/base/2021-03-06-17-12-26/validation_eval/events.out.tfevents.1615069012.dptam-comp"):
    print(summary)