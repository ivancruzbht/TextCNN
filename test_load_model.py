import model_export
import tensorflow as tf
import datetime
import numpy as np
import data_helpers
from tensorflow.contrib import learn

model_file = "/home/ivan/venuelytics/analytics/classifiers/TextCNN/runs/1481752440/checkpoints/output_graph.pb"

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("val_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")
tf.flags.DEFINE_string("vocab_dict_file", "./data/serialized-models/vocab_dict.json", "Vocab dictionary.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

#Model export parameters
tf.flags.DEFINE_string("input_graph_name", "input_graph.pb","Graph input file of the graph to export")
tf.flags.DEFINE_string("output_graph_name", "output_graph.pb","Graph output file of the graph to export")
tf.flags.DEFINE_string("output_node","output/predictions","The output node of the graph")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/validation/test set

dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
val_sample_index = dev_sample_index - int(FLAGS.val_sample_percentage * float(len(y)))
x_train, x_val, x_dev = x_shuffled[:val_sample_index], x_shuffled[val_sample_index:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_val, y_dev = y_shuffled[:val_sample_index], y_shuffled[val_sample_index:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Val/Dev split: {:d}/{:d}/{:d}".format(len(y_train), len(y_val), len(y_dev)))


def dev_step(sess, x, y, drop, loss, accuracy, x_batch, y_batch):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      x: x_batch,
      y: y_batch,
      drop: 1.0
    }
    loss, accuracy = sess.run(
        [loss, accuracy],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print("{}: loss {:g}, acc {:g}".format(time_str, loss, accuracy))


# We use our load_graph function on the file
graph = model_export.load_model(model_file)

# We can verify that we can access to the list of operations in the graph
for op in graph.get_operations():
    print(op.name)

# We access the input and output nodes 
input_x = graph.get_tensor_by_name('prefix/input_x:0')
scores = graph.get_tensor_by_name('prefix/output/scores:0')
predictions = graph.get_tensor_by_name('prefix/output/predictions:0')
drop = graph.get_tensor_by_name('prefix/dropout_keep_prob:0')


    
# We launch a Session
with tf.Session(graph=graph) as sess:
    input_y = tf.placeholder(tf.float32, [None, 2], name="input_y")
    # CalculateMean cross-entropy loss
    with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(scores, input_y)
        loss = tf.reduce_mean(losses)

    # Accuracy
    with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        
    dev_step(sess, input_x, input_y, drop, loss, accuracy, x_dev, y_dev)
    
    example = [0] * 56
    example[0] = 7080
    example[1] = 2294
    example[2] = 1776
    example[3] = 2344
    example[4] = 14041
    example[5] = 941
    example[6] = 12
    example[7] = 8186
    example[8] = 1991
    
    print(example)
    example = [example]
    print(example)
    
    print(predictions.eval(feed_dict = { input_x : example, drop:1.0}))
    
    