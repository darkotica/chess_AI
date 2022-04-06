import tensorflow as tf
import numpy as np
from tensorflow import *
from keras import *
from keras.layers import *
from keras.models import *
from keras_preprocessing import *
from tensorflow._api.v2 import data
from tensorflow.keras import *
from feature_extractor_v3 import *
from chess import *
from tensorflow.keras import backend as K
import math

# ulaz je tabla
#   768 - one hot encoding figura

batch_size_ds = 512
ds_size = 38000000


def get_dataset_partitions_tf(ds, train_split=0.8):
    
    train_size = int(int(train_split * ds_size) / batch_size_ds)
    
    print(train_size)
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size)
    
    return train_ds, val_ds

def get_feautures_from_FEN(fen):
    stringic_valjda = fen.astype(str)[0]
    board_chess = chess.Board(stringic_valjda)
    features = extract_feautre(board_chess)
    return np.asarray(features, dtype=np.float32)
    
def limit_y(y):
    yic = y[0]
    #if yic >= 30:
    #    yic = 30 + (152-yic)/152*3
    #elif y[0] <= -30:
    #    yic = -30 - (-152+yic)/152*3
      
    #yic = 2*(yic + 34)/(34 + 34) - 1
    #return tf.constant(yic)
    
    #normalizacija 0-1
    #yic = (yic  + 152.5)/(152.5 + 152.5)
    
    if yic >= 10:
        yic = 10 + math.log10(153-yic)*4
    elif yic <= -10:
        yic = -10 - math.log10(153+yic)*4
      
    yic = 2*(yic + 20)/(20 + 20) - 1
    return tf.constant(float(yic))


datasetic = tf.data.experimental.make_csv_dataset(
    'training_data/dataset_37m.csv',
    batch_size=batch_size_ds,
    label_name='Evaluations',
    num_epochs=1
)
datasetic_aug = datasetic.map(lambda x, y: (tf.numpy_function(get_feautures_from_FEN, [x['FEN']], Tout=[tf.float32]), tf.numpy_function(limit_y, [y], Tout=[tf.float32]) ))
dataset_train, dataset_val = get_dataset_partitions_tf(datasetic_aug, 0.95)

input = Input(shape=(768,))

y = Dense(4000, activation="relu")(input)
y = Dropout(0.2)(y)
y = Dense(4000, activation="relu")(y)
y = Dropout(0.2)(y)
y = Dense(4000, activation="relu")(y)
y = Dropout(0.2)(y)

y = Dense(1, activation="linear")(y)

model = Model(inputs=input, outputs=y)

model.compile(loss="mean_squared_error", optimizer="adam")
model.summary()

print("\nModel created, starting with training\n\n")

history = model.fit(x=dataset_train, validation_data=dataset_val, epochs=10)

model_json = model.to_json()
with open("model_chess_ai.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_chess_ai.h5")
print("Saved model to disk")
