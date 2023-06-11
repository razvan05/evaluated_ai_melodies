import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding
import sys 

def near_disonances_notes_model(config, notes_encoding_size, durations_encoding_size):

    # for training
    seq_len = config.sequence_length
    encoder_input = Input(shape = (seq_len, durations_encoding_size))
    
    forward_encoder = GRU(config.hidden_values, return_state = True)
    backward_encoder = GRU(config.hidden_values, return_state = True, go_backwards = True)
    f_encoder_output, f_state_h = forward_encoder(encoder_input)
    b_encoder_output, b_state_h = backward_encoder(encoder_input)

    decoder_input = Input(shape = (seq_len, notes_encoding_size))
    
    forward_decoder = GRU(config.hidden_values, return_state = True)
    backward_decoder = GRU(config.hidden_values, return_state = True, go_backwards = True)
    
    bigru_decoder = Bidirectional(forward_decoder, backward_layer = backward_decoder) 
    bigru_decoder_output, f_h, b_h = bigru_decoder(decoder_input, initial_state = [f_state_h, b_state_h])

    tf.print(bigru_decoder_output.shape, output_stream = sys.stdout)
    decoder_dense =  Dense(1, activation = 'sigmoid')
    decoder_output = decoder_dense(bigru_decoder_output)
    tf.print(decoder_output.shape, output_stream = sys.stdout)
    
    model = Model([encoder_input, decoder_input], decoder_output)

    # for prediction
    encoder_output_states = [f_state_h, b_state_h]
    encoder_model = Model(encoder_input, encoder_output_states)

    f_decoder_state_input_h = Input(shape = (config.hidden_values,))
    b_decoder_state_input_h = Input(shape = (config.hidden_values,))
    decoder_input_states = [f_decoder_state_input_h, b_decoder_state_input_h]
    
    bigru_decoder_output, f_h, b_h = bigru_decoder(decoder_input, initial_state = decoder_input_states)
    decoder_states = [f_h, b_h] 
    
    decoder_dense = decoder_dense(bigru_decoder_output)

    decoder_model = Model([decoder_input] + decoder_input_states, [decoder_dense] + decoder_states)

    opt = tf.keras.optimizers.Adam(learning_rate = config.learning_rate)
    bce = tf.keras.losses.BinaryCrossentropy()
    model.compile(loss = 'bce', optimizer = opt, metrics = ['accuracy'], run_eagerly = False)

    return model, encoder_model, decoder_model
