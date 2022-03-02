import chess

board = chess.Board()
board.san_and_push(chess.Move(chess.E2, chess.E4))

print(board)