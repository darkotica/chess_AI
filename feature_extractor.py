import chess
from chess import *


def get_matrix_pos(index):
    return [index // 8, index % 8]

def get_squares_attackers_and_defenders(board : Board):
    squares_atk_def = []
    for i in range(0, 64):
        squares_atk_def.extend(get_lowest_valued_attacker_of_piece(WHITE, i, board) 
        + get_lowest_valued_attacker_of_piece(BLACK, i, board))

    return squares_atk_def

def get_lowest_valued_attacker_of_piece(color : Color, index, board : Board): # color je boja napadaca
    attackers_mask = board.attackers_mask(color, index)
    attacker_pieces = board.piece_map(mask=attackers_mask)
    if len(attacker_pieces) == 0:
        return [-1, -1, -1, -1, -1, -1]
    # pawn, rook, knight, bishop, queen, king
    min_val_figure = min(attacker_pieces.values(), key=lambda x: x.piece_type)
    ret_list = [0, 0, 0, 0, 0, 0]
    ret_list[min_val_figure.piece_type-1] = 1
    return ret_list

def get_positional_features_of_pieces(color : Color, board : Board, piece_type : PieceType):
    bitmap_of_pieces = board.pieces(piece_type, color).tolist()

    positional_features = []
    num_of_pieces = 0
    for i in range(0, len(bitmap_of_pieces)): 
        if bitmap_of_pieces[i]:
            num_of_pieces += 1
            positional_features.extend(get_matrix_pos(i) + get_lowest_valued_attacker_of_piece(color, i, board)  # ovo ce biti ko ga brani
            + get_lowest_valued_attacker_of_piece(WHITE if color == BLACK else BLACK, i, board))  # ovo ce biti ko ga napada
            

    if (piece_type == KING):
        return positional_features

    # svake figure ima max 8 - ovo ukljucuje i pijune, ali i bas figure (sem kralja)
    # u ovakvom slucaju vodimo racuna o tome da postoji mogucnost da ima 8 komada svih figura osim kralja
    # TODO probaj da izmenis ovo kasnije, mozda da se zakuca na max 2 figure kod obicnih, sem kod kraljice i kralja
    for i in range(num_of_pieces, 8):
        positional_features.extend([-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # prva dva broja su pozicija (-1 i -1 pa ne postoji figura), a ostalo nule
        # da bi bilo neutralno
    
    return positional_features


def extract_feautre(chess_board : Board):
    features = []

    # statistic numbers

    # side to move: 1 if white, 0 if black
    features.append(1 if chess_board.turn else 0)
    # white long castle
    features.append(1 if chess_board.has_queenside_castling_rights(WHITE) else 0)
    # white short castle
    features.append(1 if chess_board.has_kingside_castling_rights(WHITE) else 0)
    # black long castle
    features.append(1 if chess_board.has_queenside_castling_rights(BLACK) else 0)
    # black short castle
    features.append(1 if chess_board.has_kingside_castling_rights(BLACK) else 0)
    # number of pieces
    features.append(len(chess_board.pieces(QUEEN, WHITE)))
    features.append(len(chess_board.pieces(ROOK, WHITE)))
    features.append(len(chess_board.pieces(BISHOP, WHITE)))
    features.append(len(chess_board.pieces(KNIGHT, WHITE)))
    features.append(len(chess_board.pieces(PAWN, WHITE)))
    features.append(len(chess_board.pieces(QUEEN, BLACK)))
    features.append(len(chess_board.pieces(ROOK, BLACK)))
    features.append(len(chess_board.pieces(BISHOP, BLACK)))
    features.append(len(chess_board.pieces(KNIGHT, BLACK)))
    features.append(len(chess_board.pieces(PAWN, BLACK)))


    # positions

    features.extend(get_positional_features_of_pieces(WHITE, chess_board, PAWN))
    features.extend(get_positional_features_of_pieces(WHITE, chess_board, ROOK))
    features.extend(get_positional_features_of_pieces(WHITE, chess_board, BISHOP))
    features.extend(get_positional_features_of_pieces(WHITE, chess_board, KNIGHT))
    features.extend(get_positional_features_of_pieces(WHITE, chess_board, QUEEN))
    features.extend(get_positional_features_of_pieces(WHITE, chess_board, KING))

    features.extend(get_positional_features_of_pieces(BLACK, chess_board, PAWN))
    features.extend(get_positional_features_of_pieces(BLACK, chess_board, ROOK))
    features.extend(get_positional_features_of_pieces(BLACK, chess_board, BISHOP))
    features.extend(get_positional_features_of_pieces(BLACK, chess_board, KNIGHT))
    features.extend(get_positional_features_of_pieces(BLACK, chess_board, QUEEN))
    features.extend(get_positional_features_of_pieces(BLACK, chess_board, KING))

    # square positional values
    features.extend(get_squares_attackers_and_defenders(chess_board))
    
    return features 

if __name__ == '__main__':
    board = chess.Board()
    extract_feautre(board)