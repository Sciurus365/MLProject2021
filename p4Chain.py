# %%
VOICE = 3
LAG = 1024
NUM_PREDICT = 133
INCLUDE_BEGINNING = True

# %%
import pandas as pd
import numpy as np
from sklearn.multioutput import RegressorChain
from sklearn.linear_model import RidgeClassifierCV
import midiutil

def predict_feature(input_df):
    l, _ = input_df.shape
    # Lagged input
    X = input_df.to_numpy()
    for lag in range(1, LAG + 1):
        next_lag = input_df[lag:]
        X = np.delete(X, l - lag, axis=0)
        X = np.append(X, next_lag, axis=1)

    # Remove y values from x
    np.delete(X, 0, axis=1)

    # Multi-output
    Y = input_df.to_numpy()

    # Use sklearn's regressorchain to fit 4 correlated regressors
    chain = RegressorChain(RidgeClassifierCV(alphas=[1e-5, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000])).fit(X, Y[LAG:])
    print(f"Score: {chain.score(X, Y[LAG:])};")
    print("Alphas used:", [est.alpha_ for est in chain.estimators_])
    # Predict some notes
    for _ in range(NUM_PREDICT):
        next_notes = chain.predict(X[-lag:])
        next_notes = next_notes.reshape(lag, 4)[-1]
        next_row = np.append(next_notes, X[-1, :-4]).reshape(1, -1)
        X = np.append(X, next_row, axis=0)

    predicted_notes = pd.DataFrame(np.round(X[-NUM_PREDICT:, 0:4]), columns=['v1', 'v2', 'v3', 'v4'])
    return predicted_notes

# Import the csv
piano_input = pd.read_csv('F.txt', sep='\t', header=None, names=['v1', 'v2', 'v3', 'v4'])
# Transform to midi values
midi_input = piano_input.where(piano_input == 0, piano_input + 8)
# Split train and test data set
piano_test = piano_input.iloc[-NUM_PREDICT:]
piano_input = piano_input.drop(piano_input.tail(NUM_PREDICT).index) 


# %%
X_pitches = piano_input % 12
X_octaves = piano_input // 12
X_rests = (piano_input == 0).astype(int)

pred_y_pitches = predict_feature(X_pitches)
pred_y_octaves = predict_feature(X_octaves)
pred_y_rests   = predict_feature(X_rests)

# %%
from dataclasses import dataclass

note_len = 0.25

if INCLUDE_BEGINNING:
    y_pitches = (piano_input % 12).append(pred_y_pitches, ignore_index=True)
    y_octaves = (piano_input // 12).append(pred_y_octaves, ignore_index=True)
    y_rests   = ((piano_input == 0).astype(int)).append(pred_y_rests, ignore_index=True)


y_pitches = y_pitches.to_numpy()
y_octaves = y_octaves.to_numpy()
y_rests   = y_rests.to_numpy()


y_pred = y_octaves*12 + y_pitches
np.savetxt("predbychain_test.txt",y_pred)

# @dataclass
# class MidiNote:
#     note: int
#     time: float
#     dura: float

# # Generate midi file
# midi = midiutil.MIDIFile(4)
# midi.addTempo(0, 0, 100)

# cur = [MidiNote((o * 12 + p) * (r * -1 + 1), 0, note_len) for p, o, r in zip(y_pitches[0], y_octaves[0], y_rests[0])]

# for time in range(1, len(y_pitches)):
#     for track in range(4):
#         if time >= len(piano_input) - 1:
#             midi.addText(track, time * note_len, "*")
#         midi_note = (y_octaves[time, track] * 12 + y_pitches[time, track]) * (y_rests[time, track] * -1 + 1)
#         if cur[track].note == midi_note:
#             cur[track].dura += note_len
#         else:
#             if cur[track].note != 0:
#                 midi.addNote(track, 0, int(cur[track].note), cur[track].time, cur[track].dura, 95)
#             cur[track] = MidiNote(midi_note, time * note_len, note_len)

# midi.addNote(track, 0, int(cur[track].note), cur[track].time, cur[track].dura, 95)

# with open('simplified_generated_with_beginning.mid', 'wb') as output:
#     midi.writeFile(output)



