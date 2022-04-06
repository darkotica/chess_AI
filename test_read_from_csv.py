from feature_extractor import *
from csv import reader
from chess import *
from math import *

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

def convert_positional_value_to_num(pos_val_orig, line, line_num):
    pos_val = pos_val_orig.replace('"', "")
    pos_val = pos_val.replace('+', "")
    pos_val = pos_val.replace('#', "")
    try:
        num = float(pos_val)
        return num
    except Exception as e:
        print(e)
        print(line)
        print(line_num)
        exit()

def get_min_max_valued_position(): 
    # open file in read mode
    with open('training_data/archive/chessData.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row in the csv using reader object
            max_position = ""
            max_value = -math.inf
            min_position = ""
            min_value = math.inf
            index = 0
            for row in csv_reader:
                if index % 1000000 == 0:
                    print("index: " + str(index))

                value = convert_positional_value_to_num(row[1], row, index)
                if value >= max_value:
                    max_value = value
                    max_position = row[0]
                if value <= min_value:
                    min_value = value
                    min_position = row[0]

                index += 1
            
            f = open("max_min_positions.txt", "w")
            f.write(max_position + " : " + str(max_value) + "\n" + min_position + " : " + str(min_value))

def get_mean(): 
    # open file in read mode
    with open('train_dataset.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row in the csv using reader object
            sum = 0
            index = 0
            for row in csv_reader:
                value = convert_positional_value_to_num(row[1], row, index)
                sum += value

                index += 1
            
            f = open("train_dataset_mean.txt", "w")
            f.write(str(sum/index))

def get_std(mean):
    with open('train_dataset.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
            # Iterate over each row in the csv using reader object
            sum = 0
            index = 0
            for row in csv_reader:
                value = convert_positional_value_to_num(row[1], row, index)
                sum += (value-mean)**2

                index += 1
            
            variance = sum / (index-1)
            std = math.sqrt(variance)
            f = open("train_dataset_mean.txt", "a")
            f.write(str(std))


get_std(44.81190610480602)