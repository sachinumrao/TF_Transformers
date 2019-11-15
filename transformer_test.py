import tensorflow as tf
from transformers import TFBertModel, BertTokenizer, TFGPT2Model, GPT2Tokenizer
import tensorflow_datasets as td
from transformers import glue_convert_examples_to_features as gcef

# loading bert model
bert_model = TFBertModel.from_pretrained("bert-base-cased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

# loading gpt model
gpt_model = TFGPT2Model.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


# load data
data = td.load("glue/mrpc")

# separate train and validation data
train = data["train"]
val = data["validation"]

# transform examples into features
train_data = gcef(train, bert_tokenizer, 128, "mrpc")
val_data = gcef(val, bert_tokenizer, 128, "mrpc")

train_data = train_data.shuffle(100).batch(32).repeat(2)
val_data = val_data.batch(64)

# define training params
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, 
                                    clipnorm=1.0)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

bert_history = bert_model.fit(train_data,
                            epochs=2,
                            steps_per_epoch=115)

