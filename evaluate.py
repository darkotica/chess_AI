from json import load
from keras.models import model_from_json
from feature_extractor_v3 import extract_feautre
from chess import *
import tensorflow as tf


def load_model(model_path, weights_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")

    loaded_model.compile(loss="mean_squared_error", optimizer="adam")
    #loaded_model.summary()

    return loaded_model

if __name__ == "__main__":
    loaded_model = load_model('working_model/model_chess_ai.json', "working_model/model_chess_ai.h5")
    
    board_chess = Board("rnbqkbnr/ppp2ppp/8/3p4/8/5BP1/PPPPPP1P/RNBQK2R w KQkq - 0 5")
    features = extract_feautre(board_chess)

    nn_input = tf.reshape(features, [1, len(features)])

    pred = loaded_model(nn_input)
    pred_f = float(pred)
    print("score: " + str(pred))
    print("score f:" + str(pred_f))
    