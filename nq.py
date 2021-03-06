import tensorflow as tf
import os
import gzip
import json
from functools import partial
import datetime
from t5.models import mesh_transformer

import tensorflow_datasets as tfds

import t5

# Public directory of Natural Questions data on GCS.
NQ_JSONL_DIR = "data/nq"

DATA_DIR = "data/nq"

NQ_SPLIT_FNAMES = {
    "train": "v1.0-simplified_simplified-nq-train.jsonl.gz",
    "validation": "v1.0-simplified_nq-dev-all.jsonl.gz"
}

nq_counts_path = os.path.join(DATA_DIR, "nq-counts.json")
nq_tsv_path = {
    "train": os.path.join(DATA_DIR, "nq-train.tsv"),
    "validation": os.path.join(DATA_DIR, "nq-validation.tsv")
}

def nq_jsonl_to_tsv(in_fname, out_fname):
    def extract_answer(tokens, span):
        """Reconstruct answer from token span and remove extra spaces."""
        start, end = span["start_token"], span["end_token"]
        ans = " ".join(tokens[start:end])
        # Remove incorrect spacing around punctuation.
        ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
        ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
        ans = ans.replace("( ", "(").replace(" )", ")")
        ans = ans.replace("`` ", "\"").replace(" ''", "\"")
        ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
        return ans

    count = 0
    with tf.io.gfile.GFile(in_fname, "rb") as infile,\
        tf.io.gfile.GFile(out_fname, "w") as outfile:
        for line in gzip.open(infile):
            ex = json.loads(line)
            # Remove any examples with more than one answer.
            if len(ex['annotations'][0]['short_answers']) != 1:
                continue
            # Questions in NQ do not include a question mark.
            question = ex["question_text"] + "?"
            answer_span = ex['annotations'][0]['short_answers'][0]
            # Handle the two document formats in NQ (tokens or text).
            if "document_tokens" in ex:
                tokens = [t["token"] for t in ex["document_tokens"]]
            elif "document_text" in ex:
                tokens = ex["document_text"].split(" ")
            answer = extract_answer(tokens, answer_span)
            # Write this line as <question>\t<answer>
            outfile.write("%s\t%s\n" % (question, answer))
            count += 1

            if count % 1000 == 0:
                print("Finished %d" % count, end='\r')

    return count

if tf.io.gfile.exists(nq_counts_path):
    # Used cached data and counts.
    print("Loading NQ from cache")
    num_nq_examples = json.load(tf.io.gfile.GFile(nq_counts_path))
else:
    # Create TSVs and get counts.
    print("Generating TSV from counts")
    num_nq_examples = {}
    for split, fname in NQ_SPLIT_FNAMES.items():
        num_nq_examples[split] = nq_jsonl_to_tsv(
            os.path.join(NQ_JSONL_DIR, fname), nq_tsv_path[split])
    json.dump(num_nq_examples, tf.io.gfile.GFile(nq_counts_path, "w"))

def nq_dataset_fn(split, shuffle_files=False):
      # We only have one file for each split.
      del shuffle_files

      # Load lines from the text file as examples.
      ds = tf.data.TextLineDataset(nq_tsv_path[split])
      # Split each "<question>\t<answer>" example into (question, answer) tuple.
      ds = ds.map(
          partial(tf.io.decode_csv, record_defaults=["", ""],
                            field_delim="\t", use_quote_delim=False),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      # Map each tuple to a {"question": ... "answer": ...} dict.
      ds = ds.map(lambda *ex: dict(zip(["question", "answer"], ex)))
      return ds

# print("A few raw validation examples...")
# for ex in tfds.as_numpy(nq_dataset_fn("validation").take(5)):
#     print(ex)

def trivia_preprocessor(ds):
    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        return text

    def to_inputs_and_targets(ex):
        """Map {"question": ..., "answer": ...}->{"inputs": ..., "targets": ...}."""
        return {
            "inputs":
                 tf.strings.join(
                     ["trivia question: ", normalize_text(ex["question"])]),
            "targets": normalize_text(ex["answer"])
        }
    return ds.map(to_inputs_and_targets,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)


t5.data.TaskRegistry.add(
    "nq_context_free",
    # Specify the task type.
    t5.data.Task,
    # Supply a function which returns a tf.data.Dataset.
    dataset_fn=nq_dataset_fn,
    splits=["train", "validation"],
    # Supply a function which preprocesses text from the tf.data.Dataset.
    text_preprocessor=[trivia_preprocessor],
    # Lowercase targets before computing metrics.
    postprocess_fn=t5.data.postprocessors.lower_text,
    # We'll use accuracy as our evaluation metric.
    metric_fns=[t5.evaluation.metrics.accuracy],
    # Not required, but helps for mixing and auto-caching.
    num_input_examples=num_nq_examples
)

nq_task = t5.data.TaskRegistry.get("nq_context_free")
ds = nq_task.get_dataset(split="validation", sequence_length={"inputs": 128, "targets": 32})

train_batch_size = 128

now = datetime.datetime.now()
ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                        now.second)
exp_dir_name = os.path.join("exp_out/nq/t5_1.1/base", ts)

if not os.path.exists(exp_dir_name):
    os.makedirs(exp_dir_name)

os.makedirs(os.path.join(exp_dir_name, "validation_eval"))

model = t5.models.MtfModel(
    tpu_job_name=None,
    tpu=None,
    gcp_project=None,
    model_dir=exp_dir_name,
    batch_size=train_batch_size,
    sequence_length={"inputs": 128, "targets": 32},
    learning_rate_schedule=0.003,
    save_checkpoints_steps=5000,
    keep_checkpoint_max=None,
    iterations_per_loop=100,
)

model.finetune(
    mixture_or_task_name="nq_context_free",
    pretrained_model_dir="pretrained_models/t5.1.1.base",
    finetune_steps=1100000
)

# model.batch_size = train_batch_size * 4
# model.eval(
#     mixture_or_task_name="nq_context_free",
#     checkpoint_steps=-1
# )