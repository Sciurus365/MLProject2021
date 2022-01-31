from dataclasses import dataclass
from sklearn.linear_model import Ridge, RidgeCV
import numpy as np
import pandas as pd
import midiutil

# Options for creating the dataframe.
# VOICE = which voice to pick, LAG = how many periods to lag for.
VOICE = 0
LAG = 128  # 128 time units ≈ 8 seconds
S_PREDICT = 20  # How many seconds to predict
num_predict = S_PREDICT * 8

def softmax(x):
    """Numerically stable softmax, from
    https://www.delftstack.com/howto/numpy/numpy-softmax/"""
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x


def topk(x, k=3):
    """Find the indices of the top-k largest values in an array"""
    inds = np.argpartition(x.flatten(), -k)[-k:]
    return np.vstack(np.unravel_index(inds, x.shape)).T


@dataclass
class Note:
    note: int = 0  # The pitch
    octave: int = 0  # The (Midi) octave
    rest: int = 0  # Whether the note is a rest
    duration: float = 0  # Just for MIDI conversion
    time: float = 0  # Just for MIDI conversion
    generated: bool = False  # Just for MIDI conversion

    def to_3d(self):
        y = np.zeros((2, 12, 11))
        y[self.rest, self.note, self.octave] = 1
        return y

    @staticmethod
    def from_3d(arr):
        topn = 1
        # Perform softmax
        soft_arr = softmax(arr)
        # Find indices of largest probabilities
        topn_i = topk(soft_arr, k=topn)
        # Put largest probabilities in array
        topn_p = [soft_arr[i[0], i[1], i[2]] for i in topn_i]
        # Normalize so probabilities sum to 1
        topn_p = topn_p / sum(topn_p)
        # Pick a note from top 3 using softmaxed values as probabilities
        i = np.random.choice(np.arange(topn), p=topn_p)
        # Take the selected index
        ind = topn_i[i]
        # Convert the indexes to a note
        n = Note(ind[1], ind[2], ind[0])
        # Set the generated boolean to true so we can show it in the sheet music
        n.generated = True
        return n


# Import the csv
piano_input = pd.read_csv('F.txt', sep='\t', header=None)

# Transform piano key pitch (A0 = 1) to Midi pitch (A0 = 21).
# The +8 was taken by comparing the generated output to the sheet music
midi_input = piano_input.where(piano_input == 0, piano_input + 8)
# Take the voice we are working on
voice_input = midi_input.iloc[:, VOICE]

# Split train and test data set
voice_test = voice_input.iloc[-num_predict:]
voice_input = voice_input.drop(voice_input.tail(num_predict).index)

# Create an output dataframe
notes = []

# Convert the notes to our feature representation
for entry in voice_input:
    note = entry % 12
    octv = entry // 12
    cur = Note(note, octv, 1 if entry == 0 else 0, 1)
    notes.append(cur)

# Convert feature representation to 3d-vector
array_rep = np.asarray([note.to_3d() for note in notes])

# Fit a linear regression to the data
n, nx, ny, nz = array_rep.shape
reshaped = array_rep.reshape((n, nx * ny * nz))
X = reshaped
for lag in range(1, LAG + 1):
    to_append = reshaped[lag:]  # Get the notes starting at index `lag`
    l, _ = X.shape
    X = np.delete(X, l - 1, axis=0)  # Delete last row
    X = np.append(X, to_append, axis=1)  # Add lagged column

y = reshaped
reg = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1],
              fit_intercept=False).fit(X=X, y=y[:-lag])
print(
    f"Cross-validated ridge regression finished. Score: {reg.score(X, y[:-lag])}"
)


for i in range(num_predict):
    print(f"\x1b[1K\r[{i + 1:04}/{num_predict}] Predicting...", end='')
    
    # Predict next note
    next_note = reg.predict(X=X[-lag:])
    next_note = next_note.reshape((lag, nx, ny, nz))[0]

    # Create new lagged row
    next_row = X[-1:, :].reshape((lag + 1, nx, ny, nz))
    next_row[0] = next_note
    next_row = next_row.reshape(1, (lag + 1) * nx * ny * nz)
    # Append new row
    X = np.append(X, next_row, axis=0)
print()

next_notes = X[-num_predict:, :(nx * ny * nz)].reshape(num_predict, nx, ny, nz)

# Convert back to original midi format
midi_notes = [Note.from_3d(arr) for arr in next_notes]

# Covert predicted notes to original inpput voice format
original_format = [n.octave * 12 + n.note - 8 * (n.rest * -1 + 1) for n in midi_notes]

# Test performance
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

y_test = voice_test
y_predict = original_format

# Parameters
MSE = mean_squared_error(y_test,y_predict)
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_predict)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_predict)
print("R2 = ", R2)

# Plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1,figsize=(7,7),dpi=300)
ax.set_title('Predition VS Test')
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--',c='k', label="1:1 line")
plt.xlabel('Test data')
plt.ylabel('Predicted data')
ax.scatter(y_test,y_predict,c = 'b',marker = 'o' , alpha = 0.1)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--',c='k', label="1:1 line")
plt.show()



