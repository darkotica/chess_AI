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
    for i in range(0, len(bitmap_of_pieces)): 
        if bitmap_of_pieces[i]:
            positional_features.extend(get_matrix_pos(i) + get_lowest_valued_attacker_of_piece(color, i, board)  # ovo ce biti ko ga brani
            + get_lowest_valued_attacker_of_piece(WHITE if color == BLACK else BLACK, i, board))  # ovo ce biti ko ga napada
            

    if (piece_type == KING):
        return positional_features

    # svake figure ima max 8 - ovo ukljucuje i pijune, ali i bas figure (sem kralja)
    for i in range(len(positional_features), 8):
        positional_features.extend([-1, -1])
    
    return positional_features


def extract_feautre(chess_board : Board):
    features = []

    # statistic numbers
    features_I = []
    # side to move: 1 if white, 0 if black
    features_I.append(1 if chess_board.turn else 0)
    # white long castle
    features_I.append(1 if chess_board.has_queenside_castling_rights(WHITE) else 0)
    # white short castle
    features_I.append(1 if chess_board.has_kingside_castling_rights(WHITE) else 0)
    # black long castle
    features_I.append(1 if chess_board.has_queenside_castling_rights(BLACK) else 0)
    # black short castle
    features_I.append(1 if chess_board.has_kingside_castling_rights(BLACK) else 0)
    # number of pieces
    features_I.append(len(chess_board.pieces(QUEEN, WHITE)))
    features_I.append(len(chess_board.pieces(ROOK, WHITE)))
    features_I.append(len(chess_board.pieces(BISHOP, WHITE)))
    features_I.append(len(chess_board.pieces(KNIGHT, WHITE)))
    features_I.append(len(chess_board.pieces(PAWN, WHITE)))
    features_I.append(len(chess_board.pieces(QUEEN, BLACK)))
    features_I.append(len(chess_board.pieces(ROOK, BLACK)))
    features_I.append(len(chess_board.pieces(BISHOP, BLACK)))
    features_I.append(len(chess_board.pieces(KNIGHT, BLACK)))
    features_I.append(len(chess_board.pieces(PAWN, BLACK)))
    # print("\nFEATURES 1, len: " + str(len(features_I)) + "\n" + str(features_I)  + "\n")


    # positions
    features_II = []

    features_II.extend(get_positional_features_of_pieces(WHITE, chess_board, PAWN))
    features_II.extend(get_positional_features_of_pieces(WHITE, chess_board, ROOK))
    features_II.extend(get_positional_features_of_pieces(WHITE, chess_board, BISHOP))
    features_II.extend(get_positional_features_of_pieces(WHITE, chess_board, KNIGHT))
    features_II.extend(get_positional_features_of_pieces(WHITE, chess_board, QUEEN))
    features_II.extend(get_positional_features_of_pieces(WHITE, chess_board, KING))

    features_II.extend(get_positional_features_of_pieces(BLACK, chess_board, PAWN))
    features_II.extend(get_positional_features_of_pieces(BLACK, chess_board, ROOK))
    features_II.extend(get_positional_features_of_pieces(BLACK, chess_board, BISHOP))
    features_II.extend(get_positional_features_of_pieces(BLACK, chess_board, KNIGHT))
    features_II.extend(get_positional_features_of_pieces(BLACK, chess_board, QUEEN))
    features_II.extend(get_positional_features_of_pieces(BLACK, chess_board, KING))
    # print("\nFEATURES 2, len: " + str(len(features_II)) + "\n" + str(features_II) + "\n")
    # print('\nfeatureII list lens')
    # dicktionary = {}
    # for el in features_II:
    #     if len(el) in dicktionary:
    #         dicktionary[len(el)] += 1
    #     else:
    #         dicktionary[len(el)] = 1
    # print(dicktionary)
    # print("\n\n")

    # square positional values
    features_III = get_squares_attackers_and_defenders(chess_board)
    # print("\nFEATURES 3, len: " + str(len(features_III)) + "\n" + str(features_III) + "\n")
    # print('\nfeatureIII list lens')
    # dicktionary = {}
    # for el in features_III:
    #     if len(el) in dicktionary:
    #         dicktionary[len(el)] += 1
    #     else:
    #         dicktionary[len(el)] = 1
    # print(dicktionary)
    # print("\n\n")
    #return [features_I, features_II, features_III]
    listica = []
    for f in features_I:
        listica.append(f)
    
    for f in features_II:
        listica.append(f)

    for f in features_III:
        listica.append(f)
    
    for _ in range(len(listica), 1231): # todo resi ovo, ovo je samo privremeno, ne vraca uvek isti
        listica.append(1)
    return listica 

if __name__ == '__main__':
    board = chess.Board()
    extract_feautre(board)