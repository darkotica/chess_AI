import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.layers import Layer
from evaluate import *
import math
import time
from collections import deque
import sys
import numpy as np
import random
import chess


def get_feautures_from_FEN(fen):
    stringic_valjda = fen.astype(str)[0]
    board_chess = chess.Board(stringic_valjda)
    features = extract_feautre(board_chess)
    return np.asarray(features, dtype=np.float32)


def limit_y(y):
    yic = y[0]
    yic = float(yic.decode("utf-8"))

    if yic >= 10:
        yic = 10 + math.log10(yic) * 4
    elif yic <= -10:
        yic = -10 - math.log10(-yic) * 4

    yic = 2 * (yic + 20) / (20 + 20) - 1
    return tf.constant(float(yic))


def find_move(board=None, model=None):
    return 0, 1


def td_alpha_giraffe(model, lambda_in_sum=0.7, iterations=1000):
    datasetic = tf.data.experimental.make_csv_dataset(
        'training_data/700mb_dataset_split/train_dataset.csv',
        batch_size=1,
        label_name='Evaluation',
        num_epochs=1,
        shuffle=True,
        shuffle_buffer_size=1000000
    )
    datasetic_aug = datasetic.map(
        lambda x, y: (
            x,
            tf.numpy_function(limit_y,
                              [y],
                              Tout=[tf.float32])
        )
    )
    errors = []
    for it in range(iterations):
        batch_of_pos = datasetic_aug.take(256)
        for el in batch_of_pos:
            fen = str(el[0]["FEN"].numpy()[0], encoding="ascii")
            board = chess.Board(fen)
            move = random.choice(list(board.legal_moves))
            board.push(move)

            error_sum = 0

            prev_eval, move = find_move()
            # make move

            for i in range(1, 12):
                pos_eval, move = find_move()
                # make move

                error_sum += (pos_eval-prev_eval)*pow(lambda_in_sum, i)

            errors.append(error_sum)
            print("opf")
        print("ppf")


def alpha_beta_pruning(board: Board, depth, alpha, beta, maximizingPlayer,
                       model, moves=None):
    # if board.is_game_over():
    #     winner = board.outcome().winner # todo videti sta za ovo
    #     if winner is None:
    #         return (0, None)  # stalemate, draw
    #
    #     if winner is True:
    #         return (1, None)  # white won
    #     else:
    #         return (-1, None)  # black won

    if depth == 0:
        features = extract_feautre(board)
        pred = model(tf.reshape(features, [1, 768]))
        return float(pred), None, pred

    if moves is None:
        moves_deque = deque()
        for m in board.legal_moves:
            if board.is_capture(m) or len(board.attacks(m.to_square)) != 0:
                moves_deque.appendleft(m)
            else:
                moves_deque.append(m)
    else:
        moves_deque = moves

    value_eager_tensor = None

    if maximizingPlayer:
        value = -math.inf
        move = None
        i = 0
        for m in moves_deque:
            i += 1
            copy_board = board.copy()
            copy_board.push(m)

            pruning_res, _, ptt = alpha_beta_pruning(copy_board, depth - 1, alpha,
                                                beta, False, model)

            if value < pruning_res:
                value = pruning_res
                value_eager_tensor = ptt
                move = m

            alpha = max(alpha, value)
            if beta <= alpha:
                break

        return value, move, value_eager_tensor

    else:
        value = math.inf
        move = None
        i = 0
        for m in moves_deque:
            i += 1
            copy_board = board.copy()
            copy_board.push(m)

            pruning_res, _, ptt = alpha_beta_pruning(copy_board, depth - 1, alpha,
                                                beta, True, model)

            if value > pruning_res:
                value = pruning_res
                move = m
                value_eager_tensor = ptt

            beta = min(beta, value)
            if beta <= alpha:
                break

        return value, move, value_eager_tensor


def initiate_alpha_beta_pruning(board, model, depth=None):
    # before starting with alpha beta pruning, we sort list of next moves, from best to worst for current player
    # this way, we improve the first depth analysis of game tree which alpha-beta pruning searches
    # and reduce calculation time by doing so

    def get_function_for_board_eval(board: Board, model):
        def funkc(move):
            copy_b = board.copy()
            copy_b.push(move)

            eval_val = float(
                model(tf.reshape(extract_feautre(copy_b), [1, 768])))
            return eval_val

        return funkc

    moves = list(board.legal_moves)
    moves = sorted(moves, key=get_function_for_board_eval(board, model),
                   reverse=board.turn)

    search_depth = depth if depth is not None else 3

    pruning_res = alpha_beta_pruning(board, search_depth - 1, -math.inf,
                                     math.inf, board.turn, model, moves)

    return pruning_res[0], pruning_res[1], pruning_res[2]


def td_alpha_full(model, alpha=1, lambda_in_sum=0.7, steps=40):
    datasetic = tf.data.experimental.make_csv_dataset(
        'training_data/700mb_dataset_split/train_dataset.csv',
        batch_size=1,
        label_name='Evaluation',
        num_epochs=1,
        shuffle=True,
        shuffle_buffer_size=1000000
    )
    datasetic_aug = datasetic.map(
        lambda x, y: (
            x,
            tf.numpy_function(limit_y,
                              [y],
                              Tout=[tf.float32])
        )
    )
    for s in range(steps):
        print("STEP: " + str(s))
        batch_of_pos = datasetic_aug.take(256)
        el_index = 0
        for el in batch_of_pos:
            print("ELEMENT: " + str(el_index))
            el_index += 1

            fen = str(el[0]["FEN"].numpy()[0], encoding="ascii")
            board = chess.Board(fen)
            move = random.choice(list(board.legal_moves))
            board.push(move)

            with tf.GradientTape(persistent=True) as tape:
                predictions = []
                for i in range(0, 2):
                    print("PREDICTION: " + str(i))
                    _, move, pos_eval = initiate_alpha_beta_pruning(board, model, 3) # depth 3
                    board.push(move)

                    predictions.append(pos_eval)


            # treba da ima m predikcija,
            #   i za svaku nam treba njen gradijent
            gradients = [tape.gradient(mod_pred, model.trainable_weights)
                         for mod_pred in predictions]

            for i in range(len(model.trainable_weights)):
                delta_weight = 0
                for t in range(len(predictions)-1):
                    inner_sum = 0
                    for k in range(t, len(predictions)-1):
                        inner_sum += (
                            (predictions[k+1] - predictions[k])
                            * math.pow(lambda_in_sum, t-k)
                            * gradients[k][i]
                        )
                    delta_weight += alpha*inner_sum

                delta_weight = -delta_weight

                layer = model.trainable_weights[i]  # Select the layer
                if delta_weight.shape[0] == 1:
                    layer.assign_sub(delta_weight[0])
                else:
                    layer.assign_sub(delta_weight)

            del tape


def td_alpha(model, alpha=1, lambda_in_sum=0.7, steps=40):

    for s in range(steps):

        pos_1_features = extract_feautre(Board(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"))
        pos_2_features = extract_feautre(Board(
            "r1bqkbnr/pppp1ppp/8/8/1n1pP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq "
            "- 0 5"))
        pos_3_features = extract_feautre(Board(
            "r1bqkbnr/pp1p1ppp/8/8/3pP1Q1/2Nn4/PPP2PPP/R1B1K2R w KQkq "
            "- 0 8"))
        pos_4_features = extract_feautre(Board(
            "Q4b1r/pp2kppp/2n2n2/4p3/8/5N2/PPPP1PPP/RNBQKB1R w KQ - 1 8"))

        with tf.GradientTape(persistent=True) as tape:
            # treba da ima m predikcija,
            #   i za svaku nam treba njen gradijent

            predictions = [model(tf.reshape(pos_1_features, [1, 768])),
                           model(tf.reshape(pos_2_features, [1, 768])),
                           model(tf.reshape(pos_3_features, [1, 768])),
                           model(tf.reshape(pos_4_features, [1, 768]))
                           ]
            gradients = [tape.gradient(mod_pred, model.trainable_weights)
                         for mod_pred in predictions]

        for i in range(len(model.trainable_weights)):
            delta_weight = 0
            for t in range(len(predictions)-1):
                inner_sum = 0
                for k in range(t, len(predictions)-1):
                    inner_sum += (
                        (predictions[k+1] - predictions[k])
                        * math.pow(lambda_in_sum, t-k)
                        * gradients[k][i]
                    )
                delta_weight += alpha*inner_sum

            delta_weight = -delta_weight

            layer = model.trainable_weights[i]  # Select the layer
            if delta_weight.shape[0] == 1:
                layer.assign_sub(delta_weight[0])
            else:
                layer.assign_sub(delta_weight)

        del tape


if __name__ == "__main__":
    model = load_model('working_model/model_chess_ai.json',
                       "working_model/model_chess_ai.h5")
    # td_alpha(model, steps=20)
    # td_alpha_giraffe(None, 0.7, 1000)
    td_alpha_full(model)


# def descend(model, steps=40, learning_rate=100.0, learning_decay=0.95):
#
#     for s in range(steps):
#
#         predictions = []
#         # treba da ima m predikcija,
#         #   i za svaku nam treba njen gradijent
#         gradients = [k.gradients(pred, model.trainable_weights)
#                      for pred in predictions]
#
#         # BOGA PITAJ STA JE OVO POSLE K GRADIENTS
#
#         # # If your target changes, you need to update the loss
#         # loss = losses.mean_squared_error(previous_pos_eval, model.output)
#         #
#         # #  ===== Symbolic Gradient =====
#         # # Tensorflow Tensor Object
#         # gradients = k.gradients(loss, model.trainable_weights)
#         #
#         # # ===== Numerical gradient =====
#         # # Numpy ndarray Objcet
#         # evaluated_gradients = sess.run(gradients, feed_dict={model.input: inputs})
#
#         # For every trainable layer in the network
#         for i in range(len(model.trainable_weights)):
#
#             layer = model.trainable_weights[i]  # Select the layer
#             layer.assign_sub(00000)
#
#
#             # # And modify it explicitly in TensorFlow
#             # sess.run(tf.assign_sub(layer, learning_rate * evaluated_gradients[i]))
#
#         # # decrease the learning rate
#         # learning_rate *= learning_decay
#         #
#         # outputs = model.predict(inputs)
#         # rmse = sqrt(mean_squared_error(targets, outputs))
#         #
#         # print("RMSE:", rmse)