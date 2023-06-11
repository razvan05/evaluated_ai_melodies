import os, sys
from music21 import converter, instrument, note, chord, stream, midi, meter, interval
import numpy as np

import bigru

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
from tqdm import tqdm
import gc

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])
    except RuntimeError as e:
        print(e)
class Config():
    model = None
    
def generate():       
    notes = []
    durations = []
    
    notes_not_disonances = []
    durations_not_disonances = []

    #define model hyperparameters
    config = Config()
    config.model = 'bigru'
    config.learning_rate = 0.0001
    config.sequence_length = 20
    config.hidden_values = 200
    config.epochs = 5
    config.batch_size = 64
    
    unique_notes = np.load('./models/unique_notes.npy', allow_pickle = True)
    unique_dur = np.load('./models/unique_dur.npy', allow_pickle = True)

    #integer encoding
    note_to_int = dict((note, i) for i, note in enumerate(unique_notes))
    dur_to_int = dict((dur, i) for i, dur in enumerate(unique_dur))
    
    encoded_notes = [note_to_int[note] for note in unique_notes]    #126 (18 scales * 7 notes)
    encoded_dur = [dur_to_int[dur] for dur in unique_dur]           #5159
    model, _, _ = bigru.near_disonances_notes_model(config, len(to_categorical(encoded_notes)), len(to_categorical(encoded_dur)))
    model.load_weights('./models/bigru/my_checkpoint')
    
    
    #read melody
    midi_file = converter.parse('melody8.mid')  
    
    m = meter.TimeSignature('4/4')
    s = stream.Score(id='mainScore')

    voice = stream.Part(id = 'part' + str(1))
    voice.append(m)
    notes_to_parse = midi_file.flat.notes
    
    notes_seq = []
    dur_seq = []
    
    good_disonances = 0
    bad_disonances = 0
    
    for element in notes_to_parse: # for each note from the current melody
        if isinstance(element, note.Note):
            current_note = element.pitch
            notes_seq.append(current_note)
            
            current_dur = element.duration.quarterLength
            dur_seq.append(current_dur)
            
            s_len = config.sequence_length
            
            if len(notes_seq) == s_len:  # seq of 20 notes
            
                # we have a real dissonance
                if interval.Interval(noteStart = notes_seq[s_len // 2 - 2], noteEnd = notes_seq[s_len // 2 - 1]).simpleName in ["m2", "M2", "TT", "m7", "M7"]: 
                    
                    #encode notes/dur for NN input
                    encoded_notes = [note_to_int[str(note)] for note in notes_seq]    #126 (18 scales * 7 notes)
                    encoded_dur = [dur_to_int.get(str(dur), 0) for dur in dur_seq]
                    
                    encoded_notes = to_categorical(encoded_notes, num_classes = len(note_to_int))   
                    encoded_dur = to_categorical(encoded_dur, num_classes = len(dur_to_int))
                    
                    #predict if there is a disonance between note 9 and 10
                    prediction = model.predict_on_batch([tf.expand_dims(encoded_dur, axis = 0), 
                                                         tf.expand_dims(encoded_notes, axis = 0)])
                    prediction = tf.squeeze(prediction, axis = 0)
                
                    if prediction[0] < 0.25: # it predicts that there is NO dissonance
                        bad_disonances += 1
                    else: # it predicts that there is a dissonance
                        good_disonances += 1
                        
                notes_seq.pop(0)
                dur_seq.pop(0)
                
    if good_disonances + bad_disonances == 0:
        print("There are no dissonances")
    else:
        print("Good disonances:  " + str(good_disonances))
        print("Bad disonances:  " + str(bad_disonances))
        print("Aesthetic score = " + str((good_disonances * good_disonances) / ((good_disonances + 9 * bad_disonances) * (good_disonances + 9 * bad_disonances))))

generate()
