import tensorflow as tf
import numpy as np
from tensorflow import *
from keras import *
from keras.layers import *
from keras.models import *
from keras_preprocessing import *
from tensorflow._api.v2 import data
from feature_extractor import *
from chess import *



def get_dataset_partitions_tf(ds, train_split=0.8):
    ds_size = 12958036  # ja ga zakucao, inace moze kao parametar
    train_size = int(train_split * ds_size)
    val_size = ds_size - train_size
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    
    return train_ds, val_ds

def get_feautures_from_FEN(fen):
    stringic_valjda = bytes.decode(fen[0].numpy())
    board_chess = chess.Board(stringic_valjda)
    features = extract_feautre(board_chess)
    #np_features = np.array(features)
    #listica = [np.asarray(feat, dtype=np.float32) for feat in features]
    #return (listica[0], listica[1], listica[2])
    return np.asarray(features, dtype=np.float32)
    #return [listica]
    #return features

def handle_y(y):
    str_data = tf.strings.regex_replace(y[0], '"', "")
    str_data = tf.strings.regex_replace(y[0], '\+', "")
    str_data = tf.strings.regex_replace(y[0], '#', "")
    return tf.strings.to_number(str_data, tf.dtypes.float32)
    #return np.asarray(tf.strings.to_number(str_data, tf.dtypes.float32), dtype=np.float32)


datasetic = tf.data.experimental.make_csv_dataset(
    'training_data/archive/chessData.csv',
    batch_size=1,
    num_epochs=1,
    label_name='Evaluation'
)
datasetic_aug = datasetic.map(lambda x, y: (tf.py_function(get_feautures_from_FEN, [x['FEN']], Tout=[tf.float32]),  handle_y(y)))
#datasetic_aug = datasetic_aug.map(lambda x,y: (tf.RaggedTensor.from_row_lengths(values=x, row_lengths=[3,2,10]), y))
datasetic_aug = datasetic_aug.map(lambda x,y: ((x[:, :15], x[:,15:448+15], x[:, 448+15:]), y))
#datasetic_aug = datasetic_aug.map(lambda x,y: ((x[:15, :], x[15:448, :], x[448:, :]), y))
dataset_train, dataset_val = get_dataset_partitions_tf(datasetic_aug)

# print("el example")
# datasetic = datasetic_aug.take(3)
# for el in datasetic.as_numpy_iterator():
#     print(len(el[0][0]))
#     print(len(el[0][1]))
#     print(len(el[0][2]))
#     print(el)
#print(list(datasetic.as_numpy_iterator()))

# shape = tf.shape(dataset_train.take(1))
# print(shape)

# prvi sloj
input_1 = Input(shape=(15,))
input_2 = Input(shape=(448,))
input_3 = Input(shape=(768,))

# drugi sloj
x_1 = Dense(1024, activation="relu")(input_1)

x_2 = Dense(1024, activation="relu")(input_2)

x_3 = Dense(1024, activation="relu")(input_3)

concat_first_two = concatenate([x_1, x_2], axis=1)
concat_double_with_third = concatenate([concat_first_two, x_3], axis=1)

y = Dense(1024, activation="relu")(concat_double_with_third)
y = Dense(512, activation="relu")(y)
y = Dense(1, activation="linear")(y)
print(y.shape)

model = Model(inputs=[input_1, input_2, input_3], outputs=y)

model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()


# combined = Concatenate(axis=1)([xm_1.output, xm_2.output, xm_3.output])
# print(combined.shape)

# y = Dense(1024, activation="relu")(combined)
# print(y.shape)
# y = Dense(512, activation="relu")(y)
# y = Dense(1, activation="linear")(y)

# model = Model(inputs=[xm_1.output, xm_2.output, xm_3.output], outputs=y)

# model.compile(loss="mean_squared_error", optimizer="adam")
# model.summary()

print("\nModel created, starting with training\n\n")

history = model.fit(x=dataset_train, validation_data=dataset_val, batch_size=1, epochs=10)

model_json = model.to_json()
with open("model_chess_ai.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_chess_ai.json.h5")
print("Saved model to disk")

# inputB = Input(shape=(128,))
# # the first branch operates on the first input

# x = Dense(4, activation="relu")(x)
# x = Model(inputs=inputA, outputs=x)
# # the second branch opreates on the second input
# y = Dense(64, activation="relu")(inputB)
# y = Dense(32, activation="relu")(y)
# y = Dense(4, activation="relu")(y)
# y = Model(inputs=inputB, outputs=y)
# # combine the output of the two branches
# combined = concatenate([x.output, y.output])
# # apply a FC layer and then a regression prediction on the
# # combined outputs
# z = Dense(2, activation="relu")(combined)
# z = Dense(1, activation="linear")(z)