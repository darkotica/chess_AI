import chess
from chess import *

# input: 15, 768, 768

def get_squares_attackers_and_defenders(board : Board):
    squares_atk_def = []
    for i in range(0, 64):
        squares_atk_def.extend(get_attackers_of_piece(WHITE, i, board) 
        + get_attackers_of_piece(BLACK, i, board))

    return squares_atk_def

def get_attackers_of_piece(color : Color, index, board : Board): # color je boja napadaca
    attackers_mask = board.attackers_mask(color, index)
    attacker_pieces = board.piece_map(mask=attackers_mask)
    if len(attacker_pieces) == 0:
        return [0, 0, 0, 0, 0, 0]
    # pawn, rook, knight, bishop, queen, king
    attackers = [x.piece_type for x in attacker_pieces.values()]
    ret_list = [0, 0, 0, 0, 0, 0]
    set_val = 1 if color == WHITE else -1

    for atk in attackers:
        ret_list[atk-1] += set_val

    return ret_list

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

    # statistic numbers

    # side to move: 1 if white, -1 if black
    features.append(1 if chess_board.turn else -1)
    # white long castle
    features.append(1 if chess_board.has_queenside_castling_rights(WHITE) else 0)
    # white short castle
    features.append(1 if chess_board.has_kingside_castling_rights(WHITE) else 0)
    # black long castle
    features.append(-1 if chess_board.has_queenside_castling_rights(BLACK) else 0)
    # black short castle
    features.append(-1 if chess_board.has_kingside_castling_rights(BLACK) else 0)
    # number of pieces
    features.append(len(chess_board.pieces(QUEEN, WHITE)))
    features.append(len(chess_board.pieces(ROOK, WHITE)))
    features.append(len(chess_board.pieces(BISHOP, WHITE)))
    features.append(len(chess_board.pieces(KNIGHT, WHITE)))
    features.append(len(chess_board.pieces(PAWN, WHITE)))
    features.append(-len(chess_board.pieces(QUEEN, BLACK)))
    features.append(-len(chess_board.pieces(ROOK, BLACK)))
    features.append(-len(chess_board.pieces(BISHOP, BLACK)))
    features.append(-len(chess_board.pieces(KNIGHT, BLACK)))
    features.append(-len(chess_board.pieces(PAWN, BLACK)))


    # positions
    features.extend(get_board_to_list(chess_board))

    # square positional values
    features.extend(get_squares_attackers_and_defenders(chess_board))
    
    return features 

if __name__ == '__main__':
    board = chess.Board("rn1qkbnr/ppp2ppp/8/3pp3/4P1b1/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 1 4")
    extract_feautre(board)
    print("ay_lmao")