import tensorflow as tf
import numpy as np
from tensorflow import *
from keras import *
from keras.layers import *
from keras.models import *
from keras_preprocessing import *
from tensorflow._api.v2 import data
from tensorflow.keras import *
from feature_extractor import *
from chess import *

# ulaz u konstelaciji toga da svih figura sem kralja moze biti po 8
#   15
#   1148
#   768
#
#Total: 1931

max_pos_val = 15319
min_pos_val = -15312


def get_dataset_partitions_tf(ds, train_split=0.8):
    ds_size = 12958036  # ja ga zakucao, inace moze kao parametar
    train_size = int(train_split * ds_size)
    val_size = ds_size - train_size
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    
    return train_ds, val_ds

def get_feautures_from_FEN(fen):
    # stringic_valjda = fen.astype(str)[0]
    # board_chess = chess.Board(stringic_valjda)
    # features = extract_feautre(board_chess)
    # return np.asarray(features, dtype=np.float32)
    stringic = str(fen)
    board_chess = chess.Board(stringic[2:len(stringic)-1])
    features = extract_feautre(board_chess)
    return np.asarray(features, dtype=np.float32)

def handle_y(y):
    str_data = tf.strings.regex_replace(y[0], '"', "")
    str_data = tf.strings.regex_replace(y[0], '\+', "")
    str_data = tf.strings.regex_replace(y[0], '#', "")
    y_val = tf.strings.to_number(str_data, tf.dtypes.float32)
    #y_val = 2*(y_val - min_pos_val)/(max_pos_val - min_pos_val) - 1     :: [-1, 1]
    #y_val = (y_val - (min_pos_val + 1000))/(max_pos_val - (min_pos_val - 1000))    # :: [0, 1], 1k je cisto da imamo praznog prostora
    return y_val


# datasetic = tf.data.experimental.make_csv_dataset(
#     'training_data/archive/chessData.csv',
#     batch_size=1024,
#     label_name='Evaluation',
#     num_epochs=1
# )
datasetic = tf.data.experimental.SqlDataset("sqlite", "training_data/test.db",
                                          "SELECT fen,eval FROM evaluations",
                                          (tf.string, tf.int32))
#datasetic = datasetic.batch(512)

datasetic_aug = datasetic.map(lambda x, y: (tf.numpy_function(get_feautures_from_FEN, [x], Tout=[tf.float32]),  y)) # izbacen handle y
datasetic_aug = datasetic_aug.map(lambda x,y: ({'input_1': x[:, :15], 'input_2': x[:,15:1148+15], 'input_3': x[:, 1148+15:]}, y))
datasetic_aug = datasetic_aug.map(lambda x,y: ({'input_1': tf.reshape(x['input_1'], [1,15]), 'input_2': tf.reshape(x['input_2'], [1,1148]), 'input_3': tf.reshape(x['input_3'], [1,768])}, tf.reshape(y, [1, 1])))
datasetic_aug = datasetic_aug.batch(512).prefetch(tf.data.AUTOTUNE)
dataset_train, dataset_val = get_dataset_partitions_tf(datasetic_aug)

#print(datasetic_aug.element_spec)
# for el in datasetic_aug:
#     print(el)

# print("el example")
# datasetic = datasetic_aug.take(20)
# for el in datasetic.as_numpy_iterator():
#     print(len(el[0][0]))
#     print(len(el[0][1]))
#     print(len(el[0][2]))
#print(list(datasetic.as_numpy_iterator()))

# shape = tf.shape(dataset_train.take(1))
# print(shape)

# prvi sloj
input_1 = Input(shape=(1,15,), name="input_1")
input_2 = Input(shape=(1,1148,), name="input_2")
input_3 = Input(shape=(1,768,), name="input_3")

# drugi sloj
x_1 = Dense(256, activation="relu")(input_1)

x_2 = Dense(2048, activation="relu")(input_2)

x_3 = Dense(1024, activation="relu")(input_3)

concat_first_two = concatenate([x_1, x_2], axis=2)
concat_double_with_third = concatenate([concat_first_two, x_3], axis=2)

y = Dense(512, activation="relu")(concat_double_with_third)
#y = Dense(512, activation="relu")(y)
y = Dense(1)(y)

model = Model(inputs=[input_1, input_2, input_3], outputs=y)

model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()

print("\nModel created, starting with training\n\n")

history = model.fit(x=datasetic_aug, validation_data=datasetic_aug, epochs=1)

model_json = model.to_json()
with open("model_chess_ai.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_chess_ai.h5")
print("Saved model to disk")
