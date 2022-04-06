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

    return loaded_model

if __name__ == "__main__":
    loaded_model = load_model('results/results_uga_buga/model_chess_ai.json', "results/results_uga_buga/model_chess_ai.h5")
    
    board_chess = Board("8/7k/6p1/3n2P1/p7/4r3/6K1/8 w - - 0 56")
    features = extract_feautre(board_chess)

    nn_input = tf.reshape(features, [1, len(features)])

    pred = loaded_model(nn_input)
    print("score: " + str(pred))
    