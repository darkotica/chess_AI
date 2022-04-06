import chess
from chess import *

# input: 768

def get_board_to_list(board : Board):
    ret_list = []
    for i in range(0, 64):
        ret_list.extend(get_piece_at(i, board))
    
    return ret_list

def get_piece_at(index, board : Board):
    listica = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    piece = board.piece_at(index)
    if not piece:
        return listica

    set_i = 1
    offset = 0
    if piece.color == BLACK:
        set_i = -1
        offset = 6
    
    
    listica[piece.piece_type-1 + offset] = set_i
    return listica


def extract_feautre(chess_board : Board):
    features = []
    # positions
    features = get_board_to_list(chess_board)
    
    return features 

if __name__ == '__main__':
    board = chess.Board("rn1qkbnr/ppp2ppp/8/3pp3/4P1b1/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 1 4")
    extract_feautre(board)
    print("ay_lmao")