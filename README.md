# Chess AI

Chess AI is a chess engine that generates the best move for a given position. It uses deep neural network for estimating the heuristics of a certain position,
as well as alpha-beta pruning for searching through the game tree.

## About the engine

### Dataset
- Dataset consists of games found on following link https://database.lichess.org/ (september 2019). Each of these games has Stockish 14 evaluation assigned to their positions, which is used as a way of supervised learning in this case. 

### Neural network
- Input of neural network is represented by an array of 768 elements. It is created by one hot encoding every field of chess board (64 fields) for every piece there is
(6 types of different pieces in chess, that could be white or black - 12 different types in total).
- Input layer is connected to a dense layer with 2048 neurons (first hidden layer). This dense layer is then connected to the first residual block, 
which is connected to the second one. Each of these residual blocks consist of 3 dense layers with 2048 neurons, each of which has ReLU activation function
at its' output and a dropout layer between them, with 20% dropout chance. 
- Second residual block is connected to output layer which consists of 1 neuron, and it has linear activation function as its output.

### Alpha-beta pruning
- Alpha-beta pruning is used to find the next best move based on the current position. In this implementation, the alghoritm sorts the positions at the depth of 0 
(based on evaluations given by neural network), in order to search through best positions first and speed up the searching process.
- At depths larger than 0, positions that are searched through first are those that include attacking or capturing another piece. This reduces the time of the search since these moves have higher chances of being the optimal moves in given positions.

## Using the application
### Requirements
- All the dependencies listed in the <em>requirements.txt</em> file have to be installed, preferably inside an enviroment created with venv specifically for this programme. 
- To install the requirements enter the following command in your terminal:
```console
pip install -r requirements.txt
``` 
### Run
- In order to run the application, you will need to unzip the contents of the file located at <em>working_model/working_model.zip</em>, in which neural network's model and weights are saved (you should unzip them in the same working_model folder). Since the github's filesize is limited at 100mb, zipping the model's weights and model seemed like a rational choice.
- To run the application run the calculate_move.py script, with FEN string of a position you would like to find the best move for, as well as a depth of game tree search (default is 4):
```console
python calculate_move.py "5k2/pp1Q4/3p1b2/5p2/r2P4/8/4PKR1/3R4 b - - 0 32" 4
```

### Output
- After the application is done with searching for the best move, you should get an output similar to this: <br><br>
 ![output](https://user-images.githubusercontent.com/58399701/165513086-d4316f77-05c8-420a-bbfe-12fa8f8be4ec.png)
- Move heuristics represents the value of position after making the recommended move (closer to 1 is winning for white, closer to -1 is winning for black, while 0 is drawn position).
- Move represents a recommended move in UCI format.
- Solve time represents number of seconds in which the application has found the recommended move.

### Results
- At the current moment, the application plays at the rank of about 1200 FIDE rated human, which is the result based on games it palyed with bots on www.chess.com (stockfish engine adjusted for said rating). 
