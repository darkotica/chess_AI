from evaluate import *
import math
import time
from collections import deque
import sys
import numpy as np

nodes_total = 0
nodes_skipped = 0
total_depth = 4
position_dict = {}


# def add finisher

def get_function_for_board_eval(board: Board, model):
    def funkc(move):
        copy_b = board.copy()
        copy_b.push(move)

        eval_val = float(model(tf.reshape(extract_feautre(copy_b), [1, 768])))
        position_dict[board.fen()] = eval_val
        return eval_val

    return funkc


def alpha_beta_pruning(board: Board, depth, alpha, beta, maximizingPlayer,
                       model, moves=None):
    global nodes_total, nodes_skipped, position_dict
    nodes_total += 1

    if board.is_game_over():
        winner = board.outcome().winner
        if winner is None:
            return (0, None)  # stalemate, draw

        if winner is True:
            return (1, None)  # white won
        else:
            return (-1, None)  # black won

    if depth == 0:
        if board.fen() in position_dict:
            # print("existing pos")
            return position_dict[board.fen()], None
        features = extract_feautre(board)
        pred = model(tf.reshape(features, [1, 768]))
        position_dict[board.fen()] = float(pred)
        return float(pred), None

    if moves is None:
        moves_deque = deque()
        for m in board.legal_moves:
            if board.is_capture(m) or len(board.attacks(m.to_square)) != 0:
                moves_deque.appendleft(m)
            else:
                moves_deque.append(m)
    else:
        moves_deque = moves

    if maximizingPlayer:
        value = -math.inf
        move = None
        i = 0
        for m in moves_deque:
            i += 1
            copy_board = board.copy()
            copy_board.push(m)

            if copy_board.fen() in position_dict:
                pruning_res = position_dict[copy_board.fen()]
            else:
                pruning_res = \
                alpha_beta_pruning(copy_board, depth - 1, alpha, beta, False,
                                   model)[0]
                position_dict[copy_board.fen()] = pruning_res

            if value < pruning_res:
                value = pruning_res
                move = m

            alpha = max(alpha, value)
            if beta <= alpha:
                nodes_skipped += board.legal_moves.count() - i
                break

        return (value, move)

    else:
        value = math.inf
        move = None
        i = 0
        for m in moves_deque:
            i += 1
            copy_board = board.copy()
            copy_board.push(m)

            if copy_board.fen() in position_dict:
                pruning_res = position_dict[copy_board.fen()]
            else:
                pruning_res = \
                alpha_beta_pruning(copy_board, depth - 1, alpha, beta, True,
                                   model)[0]
                position_dict[copy_board.fen()] = pruning_res

            if value > pruning_res:
                value = pruning_res
                move = m

            beta = min(beta, value)
            if beta <= alpha:
                nodes_skipped += board.legal_moves.count() - i
                break

        return (value, move)


def initiate_alpha_beta_pruning(board, model, depth=None):
    # before starting with alpha beta pruning, we sort list of next moves, from best to worst for current player
    # this way, we improve the first depth analysis of game tree which alpha-beta pruning searches
    # and reduce calculation time by doing so

    global nodes_total, nodes_skipped, total_depth
    moves = list(board.legal_moves)
    moves = sorted(moves, key=get_function_for_board_eval(board, model),
                   reverse=board.turn)

    search_depth = depth if depth is not None else total_depth

    pruning_res = alpha_beta_pruning(board, search_depth - 1, -math.inf,
                                     math.inf, board.turn, model, moves)

    return pruning_res[0], pruning_res[1]


def get_next_move(board, model, depth):
    global nodes_total, nodes_skipped, total_depth, position_dict
    print("\nStarted calculating")
    start = time.time()
    alpha_beta_res = initiate_alpha_beta_pruning(board, model, depth)
    end = time.time()
    print(f"Positions: {len(position_dict)}")
    print(f"Move heuristics: {alpha_beta_res[0]}")
    print("Move (UCI format): " + alpha_beta_res[1].uci())
    print(f"Solve time: {end - start}")
    print(
        f"Nodes total: {nodes_total}, nodes skipped (at least): {nodes_skipped}")


if __name__ == '__main__':
    model = load_model('working_model/model_chess_ai.json',
                       "working_model/model_chess_ai.h5")

    # fen = sys.argv[1]
    # depth = int(sys.argv[2]) if len(sys.argv) >= 3 else None # if depth is passed as argument, else its default (4)

    fen = "r2q1rk1/1pp2ppp/p2b1n2/3Pn3/2B3b1/2N1BN2/PPP2PPP/R2Q1RK1 w - - 4 11"
    depth = 5

    starting_board = Board(fen)
    get_next_move(starting_board, model, depth)
