from dataclasses import dataclass
from sklearn.linear_model import Ridge
import numpy as np
import pandas as pd
import midiutil

# Options for creating the dataframe.
# VOICE = which voice to pick, LAG = how many periods to lag for.
VOICE = 3
LAG = 16


@dataclass
class Note:
    note: int = 0  # The pitch
    octave: int = 0  # The (Midi) octave
    rest: int = 0  # Whether the note is a rest
    duration: float = 0  # Just for MIDI conversion
    time: float = 0  # Just for MIDI conversion
    generated: bool = False

    def to_3d(self):
        y = np.zeros((2, 12, 11))
        y[self.rest, self.note, self.octave] = 1
        return y

    @staticmethod
    def from_3d(arr):
        # From regressed 3d representation
        ind = list(np.unravel_index(arr.argmax(), arr.shape))
        n = Note(ind[1], ind[2], ind[0])
        n.generated = True
        return n


# Import the csv
piano_input = pd.read_csv('F.txt', sep='\t', header=None)

# Transform piano key pitch (A0 = 1) to Midi pitch (A0 = 21) Different sources
# say different numbers for A0. Might be 9 or 21. :| 9 seems to be A(-1)
# generally though. If we map octave -1 to index 0 (which is the easiest
# probably), then we need A0 = 21.
midi_input = piano_input.where(piano_input == 0, piano_input + 20)
# Take the voice we are working on
voice_input = midi_input.iloc[:, VOICE]

# Create an output dataframe
notes = []

# Process the rest of the notes
for entry in voice_input:
    note = entry % 12
    octv = entry // 12
    cur = Note(note, octv, 1 if note == 0 else 0, 1)
    notes.append(cur)

array_rep = np.asarray([note.to_3d() for note in notes])

# Fit a linear regression to the data
n, nx, ny, nz = array_rep.shape
reshaped = array_rep.reshape((n, nx * ny * nz))
X = reshaped
for lag in range(1, LAG + 1):
    to_append = reshaped[lag:]
    l, _ = X.shape
    X = np.delete(X, l - 1, axis=0)
    np.append(X, to_append, axis=1)

y = reshaped
reg = Ridge(fit_intercept=False).fit(X=X, y=y[:-lag])

num_predict = 20 * 16  # predict 20s
for i in range(num_predict):
    next_note = reg.predict(X=X[-lag:])
    next_note = next_note.reshape((lag, nx, ny, nz))[0]
    next_note = next_note.reshape(1, nx * ny * nz)
    next_row = X[-1:, :]
    next_row[0] = next_note
    np.append(X, next_row, axis=0)

next_notes = X[-num_predict:].reshape(num_predict, nx, ny, nz)

# Convert back to original format
midi_notes = [Note.from_3d(arr) for arr in next_notes]

# Merge consecutive notes
all_notes = notes + midi_notes

midi_notes_duration = []
cur = all_notes[0]
cur.duration = 0.125

for t, mn in enumerate(all_notes[1:]):
    if cur.note == mn.note and cur.octave == mn.octave:
        cur.duration += 0.125
    else:
        midi_notes_duration.append(cur)
        cur = Note(mn.note, mn.octave, mn.rest, 0.125, (t + 1) * (0.125),
                   mn.generated)

midi_notes_duration.append(cur)

# Generate midi file
midi = midiutil.MIDIFile(1)
midi.addTempo(0, 0, 120)

seen_generated = False

for mn in midi_notes_duration:
    if mn.rest == 0:
        midi.addNote(0, 0, mn.octave * 12 + mn.note, mn.time, mn.duration, 100)
        # print(mn)
        if not seen_generated and mn.generated:
            seen_generated = True
            midi.addText(0, mn.time, "GENERATED")

with open('one-hot-output.mid', 'wb') as output:
    midi.writeFile(output)
