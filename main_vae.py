from net.variational_autoencoder import DenseVAE
from net.conv_vae import ConvVAE
from net.cdc_vae import ConvDeconvVAE
import numpy as np
import tensorflow as tf
import os
import argparse
import json

#-epochs 100 -lr 3e-5 -file chroma_rolls_all.npy
# data_path = 'data/data/lyricsMidisP0'
data_path = 'data/data/lyricsMidisP0'
output_path = 'output/'
preprocessed_folder_path = "preprocessed/"
default_preprocessed_path = 'preprocessed/all_chunks' # all_chunks is the filename of the .npy

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-epochs", type=int, required = True, help = "epochs")
    parser.add_argument("-lr", type=float, required = True, help = "learning rate")
    parser.add_argument("-file", type=str, help = "preprocessed .npy file path", default= "chroma_rolls_all.npy")
    parser.add_argument("-model", type=str, help = "type of model to use (dense, conv, cdc, etc.)", default= "dense") 
    parser.add_argument("-batch", type=int, help="batch size", default=50)   
    return parser.parse_args()


if __name__ == "__main__":
    # Preprocess data and write output file as .npy
    #   Comment this line if using cached preprocessed files!
    args = parse_arguments()
    if (args.file != "default"):
        preprocessed_path = preprocessed_folder_path + args.file
    else:
        preprocessed_path = default_preprocessed_path + "_" + str(args.n)


    # Load data from .npy and train
    #TODO: automatically load only the batch size into data
    data = np.load(preprocessed_path)
    x_train = tf.constant(data)
    instrument_units = 3
    pitch_units = 12
    song_length = x_train.shape[-1]
    print(f"Loading preprocessed data from file {preprocessed_path}. Shape: {x_train.shape}")

    # Choose model for training
    model = None
    name = args.model + '-'
    if args.model == 'dense':
        model = DenseVAE(song_length= song_length,instrument_units= instrument_units,pitch_units= pitch_units,
                    learning_rate= args.lr,epochs=args.epochs,
                            hidden_dim=512,latent_size=16)
        model.compile(optimizer = model.optimizer,)
        model.build(input_shape = (1,instrument_units, pitch_units, song_length))

    elif args.model == 'conv':
        model = ConvVAE(song_length= song_length,instrument_units= instrument_units,pitch_units= pitch_units,
                    learning_rate= args.lr,epochs=args.epochs,
                            hidden_dim=512,latent_size=32)
        model.compile(optimizer = model.optimizer,)
        model.build(input_shape = (1, pitch_units, song_length, instrument_units))
        x_train = tf.transpose(x_train, perm=[0, 2, 3, 1])
    
    elif args.model == 'cdc':
        model = ConvDeconvVAE(song_length= song_length,instrument_units= instrument_units,pitch_units= pitch_units,
                    learning_rate= args.lr,epochs=args.epochs,
                            hidden_dim=512,latent_size=32)
        model.compile(optimizer = model.optimizer,)
        model.build(input_shape = (1, pitch_units, song_length, instrument_units))
        x_train = tf.transpose(x_train, perm=[0, 2, 3, 1])  
    
    # Train model
    model.summary()
    history = model.fit(x_train, x_train,epochs=model.epochs,batch_size = args.batch)

    # Save model and training history
    MODEL_DIR = 'saved_model/'
    HIST_DIR = 'saved_model/history/'
    print('Saving model')
    model.save(MODEL_DIR + name + 'vae')
    print('Saving model history')
    out_file = open(HIST_DIR + name + 'vae.json', "w") 
    json.dump(history.history, out_file)