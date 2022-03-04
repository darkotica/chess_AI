from feature_extractor import *
from csv import reader
from chess import *

def read_from_dataset_csv(): 
    # open file in read mode
    with open('training_data/archive/chessData.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row in the csv using reader object
            index = 0
            for row in csv_reader:
                board_chess = chess.Board(row[0])
                print("\n\nindex: " + str(index))
                if (index == 13):
                    print("opa evo ga")
                features = extract_feautre(board_chess)
                index += 1


read_from_dataset_csv()