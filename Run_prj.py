import os
from os.path import join
import argparse


PATH = os.path.dirname(__file__)

# PARSING ARGUMENTS

parser = argparse.ArgumentParser(description='Description')

parser.add_argument('-b', action="store", default=32, type=int,dest='batch_size',help='Size of the batch.')
parser.add_argument('-e', action="store",default=10,type=int,dest='epochs',help='Number of epochs')
parser.add_argument('-f', action="store", default=False, type=bool,dest='horizontal_flip',help='Set horizontal flip or not [True|False]')
parser.add_argument('-n', action="store", default=0, type=int,dest='n_layers_trainable',help='Set the number of last trainable layers')
parser.add_argument('-d', action="store", default=0, type=float,dest='dropout_rate',help='Set the dropout_rate')


parser.add_argument('--distortions', action="store", type=float,dest='disto',default=0.,help='Activate distortions or not')

parser.add_argument('--train_path', action="store", default=join(PATH, '../data/wikipaintings_10/wikipaintings_train'),dest='training_path',help='Path of the training data directory')
parser.add_argument('--val_path', action="store", default=join(PATH, '../data/wikipaintings_10/wikipaintings_val'),dest='validation_path',help='Path of the validation data directory')



args = parser.parse_args()

model_name = args.model_name
batch_size = args.batch_size
epochs = args.epochs
flip = args.horizontal_flip
TRAINING_PATH = args.training_path
VAL_PATH = args.validation_path
n_layers_trainable = args.n_layers_trainable
dropout_rate = args.dropout_rate

params = vars(args)