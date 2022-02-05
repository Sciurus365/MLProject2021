# %% [markdown]
# # Preprocessing

# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import re
import random
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, Ridge, RidgeCV
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, TransformedTargetRegressor

# %%
# from `input.ipynb`

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Helper class
@dataclass
class Note:
    note: int = 0       # The pitch
    octave: int = 0     # The (Midi) octave
    rest: int = 0       # Whether the note is a rest
    duration: int = 0   # The duration of the note (in 1/16 second intervals)

# Options for creating the dataframe. VOICE = which voice to pick, LAG = how many periods to lag for.
VOICE = 3
LAG = 10
NUM_PREDICT = 133

# Import the csv
piano_input = pd.read_csv('F.txt', sep='\t', header=None)

# Transform piano key pitch (A0 = 1) to Midi pitch (A0 = 21)
# Different sources say different numbers for A0. Might be 9 or 21. :| 9 seems to be A(-1) generally though.
# If we map octave -1 to index 0 (which is the easiest probably), then we need A0 = 21.
midi_input = piano_input.where(piano_input == 0, piano_input + 8)
# Take the voice we are working on
voice_input = midi_input.iloc[:, VOICE]

# Split test and train data set
piano_test_3 = voice_input.iloc[-NUM_PREDICT:]
voice_input = voice_input.drop(voice_input.tail(NUM_PREDICT).index) 

# Create an output dataframe
notes = []
# Process the first note
cur = Note(voice_input[0] % 12, voice_input[0] // 12, 1 if voice_input[0] == 0 else 0, 1)

# Process the rest of the notes
for entry in voice_input[1:]:
    note = entry % 12
    octv = entry // 12
    if cur.note == note and cur.octave == octv:
        cur.duration += 1
    else:
        notes.append(cur)
        cur = Note(note, octv, 1 if note == 0 and octv == 0 else 0, 1)

notes.append(cur)
cur = Note(note, octv, 1 if note == 0 and octv == 0 else 0, 1)

# Pandas can automatically convert a list of dataclass objects to dataframes!
notes_df = pd.DataFrame(notes)

# Create a lagged input dataframe
notes_lagged_df = notes_df.copy()

for lag in range(1, LAG + 1):
    lagged = notes_df.shift(lag)
    lagged.columns = [f'{col_name}_lag{lag}' for col_name in notes_df.columns]
    notes_lagged_df = pd.concat((notes_lagged_df, lagged), axis=1)

# Drop rows containing NA (i.e. the first LAG rows basically)
dfV3_full = notes_lagged_df
notes_lagged_df = notes_lagged_df.dropna()

# Show first 10 notes
notes_lagged_df.head(10)

# %% [markdown]
# # Models

# %%
dfV3_full

# %% [markdown]
# Change here: duration also as categorical

# %%
dfV3 = notes_lagged_df
# columns as categories or contineous variables
cate_cols = [i for i in dfV3.columns.values if (re.search('note', i) or re.search('rest', i) or re.search('duration', i))]
cont_cols = [i for i in dfV3.columns.values if (re.search('octave', i))]
print(cate_cols)
print(cont_cols)

# %% [markdown]
# ## model for `rest`

# %%
rcv_pipe_rest = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'rest' and i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
rcv_pipe_rest.fit(X = dfV3[dfV3.columns.drop(['rest', 'note', 'octave', 'duration'])], y = dfV3['rest'])
rcv_pipe_rest['ridgeclassifiercv'].alpha_

# %%
rcv_pipe_rest_pred = rcv_pipe_rest.predict(X = dfV3[dfV3.columns.drop(['rest', 'note', 'octave', 'duration'])])
plt.scatter(dfV3.rest, rcv_pipe_rest_pred, alpha = 0.01)
# it never predicts a rest, which is fine?
# then we don't need to predict duration for rests.

# %% [markdown]
# ## model for `octave`

# %%
rcv_pipe_octave = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000.,], cv = 10)
)
## here we need to filter out rests
rcv_pipe_octave.fit(X = dfV3[dfV3.rest != 1][dfV3.columns.drop(['note', 'octave', 'duration'])], 
                    y = dfV3[dfV3.rest != 1]['octave'])
rcv_pipe_octave['ridgeclassifiercv'].alpha_

# %%
rcv_pipe_octave_pred = rcv_pipe_octave.predict(X = dfV3[dfV3.rest != 1][dfV3.columns.drop(['note', 'octave', 'duration'])])
plt.scatter(dfV3[dfV3.rest != 1]['octave'], rcv_pipe_octave_pred, alpha = 0.01)

# %% [markdown]
# ## model for `note`

# %%
rcv_pipe_note = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_note.fit(X = dfV3[dfV3.rest != 1][dfV3.columns.drop(['note', 'duration'])], 
                  y = dfV3[dfV3.rest != 1]['note'])
rcv_pipe_note['ridgeclassifiercv'].alpha_

# %%
rcv_pipe_note_pred = rcv_pipe_note.predict(X = dfV3[dfV3.rest != 1][dfV3.columns.drop(['note', 'duration'])])
plt.scatter(dfV3[dfV3.rest != 1]['note'], rcv_pipe_note_pred, alpha = 0.01)

# %% [markdown]
# ## model for `duration`

# %%
rcv_pipe_duration = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_duration.fit(X = dfV3[dfV3.rest != 1][dfV3.columns.drop(['duration'])], 
                    y = dfV3[dfV3.rest != 1]['duration'])
rcv_pipe_note['ridgeclassifiercv'].alpha_

# %%
rcv_pipe_duration_pred = rcv_pipe_duration.predict(X = dfV3[dfV3.rest != 1][dfV3.columns.drop(['duration'])])
plt.scatter(dfV3[dfV3.rest != 1]['duration'], rcv_pipe_duration_pred, alpha = 0.01)
# this looks better.

# %% [markdown]
# # Prediction
# stochastic
dfV3_pred = dfV3_full
random.seed(1614)
for i in range(100):
    # making the next row with blanks
    temp_next_row = pd.DataFrame([np.nan, np.nan, np.nan, np.nan]+dfV3_pred.tail(1).iloc[:,:-4].values.tolist()[0][:]).T
    temp_next_row.columns = dfV3_pred.columns
    dfV3_pred = dfV3_pred.append(temp_next_row, ignore_index=True)
    
    # predict the next rest (not actually predicting)
    temp_v1_rest = 0
    dfV3_pred.iloc[-1,2] = temp_v1_rest
    # predict the next octave
    temp_v1_octave = rcv_pipe_octave.predict(X=dfV3_pred.tail(1).drop(columns = ['octave', 'note', 'duration']))
    dfV3_pred.iloc[-1,1] = temp_v1_octave
    # predict the next note
    # temp_v1_note = rcv_pipe_note.predict(X=dfV3_pred.tail(1).drop(columns = ['note', 'duration']))
    d = rcv_pipe_note.decision_function(X=dfV3_pred.tail(1).drop(columns = ['note', 'duration']))
    probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
    temp_v1_note = random.choices(rcv_pipe_note.classes_, weights=probs[0])[0]
    dfV3_pred.iloc[-1,0] = temp_v1_note
    # predict the next duration
    # temp_v1_duration = rcv_pipe_duration.predict(X=dfV3_pred.tail(1).drop(columns = ['duration']))
    d = rcv_pipe_duration.decision_function(X=dfV3_pred.tail(1).drop(columns = ['duration']))
    probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
    temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
    while(temp_v1_duration > 12):
        temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
    dfV3_pred.iloc[-1,3] = temp_v1_duration

# %%
dfV3_pred.tail(10)
# duration too long

# %%
len(dfV3_pred)
len(dfV3)


# # %% [markdown]
# # ## The same procedure, but for Little Star

# # %%
# # from `input.ipynb`

# from dataclasses import dataclass

# import numpy as np
# import pandas as pd

# # Helper class
# @dataclass
# class Note:
#     note: int = 0       # The pitch
#     octave: int = 0     # The (Midi) octave
#     rest: int = 0       # Whether the note is a rest
#     duration: int = 0   # The duration of the note (in 1/16 second intervals)

# # Options for creating the dataframe. VOICE = which voice to pick, LAG = how many periods to lag for.
# VOICE = 0
# LAG = 10

# # Import the csv
# piano_input = pd.read_csv('twinkle.txt', sep='\t', header=None)

# # Transform piano key pitch (A0 = 1) to Midi pitch (A0 = 21)
# # Different sources say different numbers for A0. Might be 9 or 21. :| 9 seems to be A(-1) generally though.
# # If we map octave -1 to index 0 (which is the easiest probably), then we need A0 = 21.
# midi_input = piano_input.where(piano_input == 0, piano_input - 12)
# # Take the voice we are working on
# voice_input = midi_input.iloc[:, VOICE]

# # Create an output dataframe
# notes = []
# # Process the first note
# cur = Note(voice_input[0] % 12, voice_input[0] // 12, 1 if voice_input[0] == 0 else 0, 1)

# # Process the rest of the notes
# for entry in voice_input[1:]:
#     note = entry % 12
#     octv = entry // 12
#     if cur.note == note and cur.octave == octv:
#         cur.duration += 1
#     else:
#         notes.append(cur)
#         cur = Note(note, octv, 1 if note == 0 and octv == 0 else 0, 1)

# notes.append(cur)
# cur = Note(note, octv, 1 if note == 0 and octv == 0 else 0, 1)

# # Pandas can automatically convert a list of dataclass objects to dataframes!
# notes_df = pd.DataFrame(notes)

# # Create a lagged input dataframe
# notes_lagged_df = notes_df.copy()

# for lag in range(1, LAG + 1):
#     lagged = notes_df.shift(lag)
#     lagged.columns = [f'{col_name}_lag{lag}' for col_name in notes_df.columns]
#     notes_lagged_df = pd.concat((notes_lagged_df, lagged), axis=1)

# # Drop rows containing NA (i.e. the first LAG rows basically)
# dfV0_twinkle_full = notes_lagged_df
# notes_lagged_df = notes_lagged_df.dropna()

# # Show first 10 notes
# notes_lagged_df.head(10)

# # %%
# dfV0_twinkle_full.head(10)

# # %%
# # stochastic
# dfV0_twinkle = dfV0_twinkle_full
# random.seed(1614)
# for i in range(100):
#     # making the next row with blanks
#     temp_next_row = pd.DataFrame([np.nan, np.nan, np.nan, np.nan]+dfV0_twinkle.tail(1).iloc[:,:-4].values.tolist()[0][:]).T
#     temp_next_row.columns = dfV0_twinkle.columns
#     dfV0_twinkle = dfV0_twinkle.append(temp_next_row, ignore_index=True)
    
#     # predict the next rest (not actually predicting)
#     temp_v1_rest = 0
#     dfV0_twinkle.iloc[-1,2] = temp_v1_rest
#     # predict the next octave
#     temp_v1_octave = rcv_pipe_octave.predict(X=dfV0_twinkle.tail(1).drop(columns = ['octave', 'note', 'duration']))
#     dfV0_twinkle.iloc[-1,1] = temp_v1_octave
#     # predict the next note
#     # temp_v1_note = rcv_pipe_note.predict(X=dfV0_twinkle.tail(1).drop(columns = ['note', 'duration']))
#     d = rcv_pipe_note.decision_function(X=dfV0_twinkle.tail(1).drop(columns = ['note', 'duration']))
#     probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
#     temp_v1_note = random.choices(rcv_pipe_note.classes_, weights=probs[0])[0]
#     dfV0_twinkle.iloc[-1,0] = temp_v1_note
#     # predict the next duration
#     # temp_v1_duration = rcv_pipe_duration.predict(X=dfV0_twinkle.tail(1).drop(columns = ['duration']))
#     d = 3.5*rcv_pipe_duration.decision_function(X=dfV0_twinkle.tail(1).drop(columns = ['duration']))
#     probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
#     temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
#     while(temp_v1_duration > 12):
#         temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
#     dfV0_twinkle.iloc[-1,3] = temp_v1_duration

# %% [markdown]
# # (Optional) the same procedure, but for other voices

# %% [markdown]
# for `Voice = 0`

# %%
# from `input.ipynb`

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Helper class
@dataclass
class Note:
    note: int = 0       # The pitch
    octave: int = 0     # The (Midi) octave
    rest: int = 0       # Whether the note is a rest
    duration: int = 0   # The duration of the note (in 1/16 second intervals)

# Options for creating the dataframe. VOICE = which voice to pick, LAG = how many periods to lag for.
VOICE = 0
LAG = 10

# Import the csv
piano_input = pd.read_csv('F.txt', sep='\t', header=None)

# Transform piano key pitch (A0 = 1) to Midi pitch (A0 = 21)
# Different sources say different numbers for A0. Might be 9 or 21. :| 9 seems to be A(-1) generally though.
# If we map octave -1 to index 0 (which is the easiest probably), then we need A0 = 21.
midi_input = piano_input.where(piano_input == 0, piano_input + 8)
# Take the voice we are working on
voice_input = midi_input.iloc[:, VOICE]

# Split test and train data set
piano_test_0 = voice_input.iloc[-NUM_PREDICT:]
voice_input = voice_input.drop(voice_input.tail(NUM_PREDICT).index) 

# Create an output dataframe
notes = []
# Process the first note
cur = Note(voice_input[0] % 12, voice_input[0] // 12, 1 if voice_input[0] == 0 else 0, 1)

# Process the rest of the notes
for entry in voice_input[1:]:
    note = entry % 12
    octv = entry // 12
    if cur.note == note and cur.octave == octv:
        cur.duration += 1
    else:
        notes.append(cur)
        cur = Note(note, octv, 1 if note == 0 and octv == 0 else 0, 1)

notes.append(cur)
cur = Note(note, octv, 1 if note == 0 and octv == 0 else 0, 1)

# Pandas can automatically convert a list of dataclass objects to dataframes!
notes_df = pd.DataFrame(notes)

# Create a lagged input dataframe
notes_lagged_df = notes_df.copy()

for lag in range(1, LAG + 1):
    lagged = notes_df.shift(lag)
    lagged.columns = [f'{col_name}_lag{lag}' for col_name in notes_df.columns]
    notes_lagged_df = pd.concat((notes_lagged_df, lagged), axis=1)

# Drop rows containing NA (i.e. the first LAG rows basically)
dfV0_full = notes_lagged_df
notes_lagged_df = notes_lagged_df.dropna()

# Show first 10 notes
notes_lagged_df.head(10)

dfV0 = notes_lagged_df
# columns as categories or contineous variables
cate_cols = [i for i in dfV0.columns.values if (re.search('note', i) or re.search('rest', i) or re.search('duration', i))]
cont_cols = [i for i in dfV0.columns.values if (re.search('octave', i))]
print(cate_cols)
print(cont_cols)

rcv_pipe_rest = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'rest' and i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
rcv_pipe_rest.fit(X = dfV0[dfV0.columns.drop(['rest', 'note', 'octave', 'duration'])], y = dfV0['rest'])
rcv_pipe_rest['ridgeclassifiercv'].alpha_

rcv_pipe_rest_pred = rcv_pipe_rest.predict(X = dfV0[dfV0.columns.drop(['note', 'rest', 'octave', 'duration'])])
plt.scatter(dfV0.rest, rcv_pipe_rest_pred, alpha = 0.01)

rcv_pipe_octave = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_octave.fit(X = dfV0[dfV0.rest != 1][dfV0.columns.drop(['note', 'octave', 'duration'])], 
                    y = dfV0[dfV0.rest != 1]['octave'])
rcv_pipe_octave['ridgeclassifiercv'].alpha_

rcv_pipe_octave_pred = rcv_pipe_octave.predict(X = dfV0[dfV0.rest != 1][dfV0.columns.drop(['note', 'octave', 'duration'])])
plt.scatter(dfV0[dfV0.rest != 1]['octave'], rcv_pipe_octave_pred, alpha = 0.01)

rcv_pipe_note = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_note.fit(X = dfV0[dfV0.rest != 1][dfV0.columns.drop(['note', 'duration'])], 
                  y = dfV0[dfV0.rest != 1]['note'])
rcv_pipe_note['ridgeclassifiercv'].alpha_

rcv_pipe_note_pred = rcv_pipe_note.predict(X = dfV0[dfV0.rest != 1][dfV0.columns.drop(['note', 'duration'])])
plt.scatter(dfV0[dfV0.rest != 1]['note'], rcv_pipe_note_pred, alpha = 0.01)

rcv_pipe_duration = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_duration.fit(X = dfV0[dfV0.rest != 1][dfV0.columns.drop(['duration'])], 
                    y = dfV0[dfV0.rest != 1]['duration'])
rcv_pipe_note['ridgeclassifiercv'].alpha_

rcv_pipe_duration_pred = rcv_pipe_duration.predict(X = dfV0[dfV0.rest != 1][dfV0.columns.drop(['duration'])])
plt.scatter(dfV0[dfV0.rest != 1]['duration'], rcv_pipe_duration_pred, alpha = 0.01)
# this looks better.

# stochastic
dfV0_pred = dfV0_full
random.seed(1614)
for i in range(100):
    # making the next row with blanks
    temp_next_row = pd.DataFrame([np.nan, np.nan, np.nan, np.nan]+dfV0_pred.tail(1).iloc[:,:-4].values.tolist()[0][:]).T
    temp_next_row.columns = dfV0_pred.columns
    dfV0_pred = dfV0_pred.append(temp_next_row, ignore_index=True)
    
    # predict the next rest (not actually predicting)
    temp_v1_rest = 0
    dfV0_pred.iloc[-1,2] = temp_v1_rest
    # predict the next octave
    temp_v1_octave = rcv_pipe_octave.predict(X=dfV0_pred.tail(1).drop(columns = ['octave', 'note', 'duration']))
    dfV0_pred.iloc[-1,1] = temp_v1_octave
    # predict the next note
    # temp_v1_note = rcv_pipe_note.predict(X=dfV0_pred.tail(1).drop(columns = ['note', 'duration']))
    d = rcv_pipe_note.decision_function(X=dfV0_pred.tail(1).drop(columns = ['note', 'duration']))
    probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
    temp_v1_note = random.choices(rcv_pipe_note.classes_, weights=probs[0])[0]
    dfV0_pred.iloc[-1,0] = temp_v1_note
    # predict the next duration
    # temp_v1_duration = rcv_pipe_duration.predict(X=dfV0_pred.tail(1).drop(columns = ['duration']))
    d = rcv_pipe_duration.decision_function(X=dfV0_pred.tail(1).drop(columns = ['duration']))
    probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
    temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
    while(temp_v1_duration > 12):
        temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
    dfV0_pred.iloc[-1,3] = temp_v1_duration

# %% [markdown]
# `Voice = 1`

# %%
# from `input.ipynb`

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Helper class
@dataclass
class Note:
    note: int = 0       # The pitch
    octave: int = 0     # The (Midi) octave
    rest: int = 0       # Whether the note is a rest
    duration: int = 0   # The duration of the note (in 1/16 second intervals)

# Options for creating the dataframe. VOICE = which voice to pick, LAG = how many periods to lag for.
VOICE = 1
LAG = 10

# Import the csv
piano_input = pd.read_csv('F.txt', sep='\t', header=None)

# Transform piano key pitch (A0 = 1) to Midi pitch (A0 = 21)
# Different sources say different numbers for A0. Might be 9 or 21. :| 9 seems to be A(-1) generally though.
# If we map octave -1 to index 0 (which is the easiest probably), then we need A0 = 21.
midi_input = piano_input.where(piano_input == 0, piano_input + 8)
# Take the voice we are working on
voice_input = midi_input.iloc[:, VOICE]

# Split test and train data set
piano_test_1 = voice_input.iloc[-NUM_PREDICT:]
voice_input = voice_input.drop(voice_input.tail(NUM_PREDICT).index) 

# Create an output dataframe
notes = []
# Process the first note
cur = Note(voice_input[0] % 12, voice_input[0] // 12, 1 if voice_input[0] == 0 else 0, 1)

# Process the rest of the notes
for entry in voice_input[1:]:
    note = entry % 12
    octv = entry // 12
    if cur.note == note and cur.octave == octv:
        cur.duration += 1
    else:
        notes.append(cur)
        cur = Note(note, octv, 1 if note == 0 and octv == 0 else 0, 1)

notes.append(cur)
cur = Note(note, octv, 1 if note == 0 and octv == 0 else 0, 1)

# Pandas can automatically convert a list of dataclass objects to dataframes!
notes_df = pd.DataFrame(notes)

# Create a lagged input dataframe
notes_lagged_df = notes_df.copy()

for lag in range(1, LAG + 1):
    lagged = notes_df.shift(lag)
    lagged.columns = [f'{col_name}_lag{lag}' for col_name in notes_df.columns]
    notes_lagged_df = pd.concat((notes_lagged_df, lagged), axis=1)

# Drop rows containing NA (i.e. the first LAG rows basically)
dfV1_full = notes_lagged_df
notes_lagged_df = notes_lagged_df.dropna()

# Show first 10 notes
notes_lagged_df.head(10)

dfV1 = notes_lagged_df
# columns as categories or contineous variables
cate_cols = [i for i in dfV1.columns.values if (re.search('note', i) or re.search('rest', i) or re.search('duration', i))]
cont_cols = [i for i in dfV1.columns.values if (re.search('octave', i))]
print(cate_cols)
print(cont_cols)

rcv_pipe_rest = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'rest' and i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
rcv_pipe_rest.fit(X = dfV1[dfV1.columns.drop(['rest', 'note', 'octave', 'duration'])], y = dfV1['rest'])
rcv_pipe_rest['ridgeclassifiercv'].alpha_

rcv_pipe_rest_pred = rcv_pipe_rest.predict(X = dfV1[dfV1.columns.drop(['note', 'rest', 'octave', 'duration'])])
plt.scatter(dfV1.rest, rcv_pipe_rest_pred, alpha = 0.01)

rcv_pipe_octave = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_octave.fit(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['note', 'octave', 'duration'])], 
                    y = dfV1[dfV1.rest != 1]['octave'])
rcv_pipe_octave['ridgeclassifiercv'].alpha_

rcv_pipe_octave_pred = rcv_pipe_octave.predict(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['note', 'octave', 'duration'])])
plt.scatter(dfV1[dfV1.rest != 1]['octave'], rcv_pipe_octave_pred, alpha = 0.01)

rcv_pipe_note = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_note.fit(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['note', 'duration'])], 
                  y = dfV1[dfV1.rest != 1]['note'])
rcv_pipe_note['ridgeclassifiercv'].alpha_

rcv_pipe_note_pred = rcv_pipe_note.predict(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['note', 'duration'])])
plt.scatter(dfV1[dfV1.rest != 1]['note'], rcv_pipe_note_pred, alpha = 0.01)

rcv_pipe_duration = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_duration.fit(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['duration'])], 
                    y = dfV1[dfV1.rest != 1]['duration'])
rcv_pipe_note['ridgeclassifiercv'].alpha_

rcv_pipe_duration_pred = rcv_pipe_duration.predict(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['duration'])])
plt.scatter(dfV1[dfV1.rest != 1]['duration'], rcv_pipe_duration_pred, alpha = 0.01)
# this looks better.

# stochastic
dfV1_pred = dfV1_full
random.seed(1614)
for i in range(100):
    # making the next row with blanks
    temp_next_row = pd.DataFrame([np.nan, np.nan, np.nan, np.nan]+dfV1_pred.tail(1).iloc[:,:-4].values.tolist()[0][:]).T
    temp_next_row.columns = dfV1_pred.columns
    dfV1_pred = dfV1_pred.append(temp_next_row, ignore_index=True)
    
    # predict the next rest (not actually predicting)
    temp_v1_rest = 0
    dfV1_pred.iloc[-1,2] = temp_v1_rest
    # predict the next octave
    temp_v1_octave = rcv_pipe_octave.predict(X=dfV1_pred.tail(1).drop(columns = ['octave', 'note', 'duration']))
    dfV1_pred.iloc[-1,1] = temp_v1_octave
    # predict the next note
    # temp_v1_note = rcv_pipe_note.predict(X=dfV1_pred.tail(1).drop(columns = ['note', 'duration']))
    d = rcv_pipe_note.decision_function(X=dfV1_pred.tail(1).drop(columns = ['note', 'duration']))
    probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
    temp_v1_note = random.choices(rcv_pipe_note.classes_, weights=probs[0])[0]
    dfV1_pred.iloc[-1,0] = temp_v1_note
    # predict the next duration
    # temp_v1_duration = rcv_pipe_duration.predict(X=dfV1_pred.tail(1).drop(columns = ['duration']))
    d = rcv_pipe_duration.decision_function(X=dfV1_pred.tail(1).drop(columns = ['duration']))
    probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
    temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
    while(temp_v1_duration > 12):
        temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
    dfV1_pred.iloc[-1,3] = temp_v1_duration

# %% [markdown]
# `Voice = 2`

# %%
# from `input.ipynb`

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Helper class
@dataclass
class Note:
    note: int = 0       # The pitch
    octave: int = 0     # The (Midi) octave
    rest: int = 0       # Whether the note is a rest
    duration: int = 0   # The duration of the note (in 1/16 second intervals)

# Options for creating the dataframe. VOICE = which voice to pick, LAG = how many periods to lag for.
VOICE = 2
LAG = 10

# Import the csv
piano_input = pd.read_csv('F.txt', sep='\t', header=None)

# Transform piano key pitch (A0 = 1) to Midi pitch (A0 = 21)
# Different sources say different numbers for A0. Might be 9 or 21. :| 9 seems to be A(-1) generally though.
# If we map octave -1 to index 0 (which is the easiest probably), then we need A0 = 21.
midi_input = piano_input.where(piano_input == 0, piano_input + 8)
# Take the voice we are working on
voice_input = midi_input.iloc[:, VOICE]

# Split test and train data set
piano_test_2 = voice_input.iloc[-NUM_PREDICT:]
voice_input = voice_input.drop(voice_input.tail(NUM_PREDICT).index) 

# Create an output dataframe
notes = []
# Process the first note
cur = Note(voice_input[0] % 12, voice_input[0] // 12, 1 if voice_input[0] == 0 else 0, 1)

# Process the rest of the notes
for entry in voice_input[1:]:
    note = entry % 12
    octv = entry // 12
    if cur.note == note and cur.octave == octv:
        cur.duration += 1
    else:
        notes.append(cur)
        cur = Note(note, octv, 1 if note == 0 and octv == 0 else 0, 1)

notes.append(cur)
cur = Note(note, octv, 1 if note == 0 and octv == 0 else 0, 1)

# Pandas can automatically convert a list of dataclass objects to dataframes!
notes_df = pd.DataFrame(notes)

# Create a lagged input dataframe
notes_lagged_df = notes_df.copy()

for lag in range(1, LAG + 1):
    lagged = notes_df.shift(lag)
    lagged.columns = [f'{col_name}_lag{lag}' for col_name in notes_df.columns]
    notes_lagged_df = pd.concat((notes_lagged_df, lagged), axis=1)

# Drop rows containing NA (i.e. the first LAG rows basically)
dfV2_full = notes_lagged_df
notes_lagged_df = notes_lagged_df.dropna()

# Show first 10 notes
notes_lagged_df.head(10)

dfV2 = notes_lagged_df
# columns as categories or contineous variables
cate_cols = [i for i in dfV2.columns.values if (re.search('note', i) or re.search('rest', i) or re.search('duration', i))]
cont_cols = [i for i in dfV2.columns.values if (re.search('octave', i))]
print(cate_cols)
print(cont_cols)

rcv_pipe_rest = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'rest' and i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
rcv_pipe_rest.fit(X = dfV2[dfV2.columns.drop(['rest', 'note', 'octave', 'duration'])], y = dfV2['rest'])
rcv_pipe_rest['ridgeclassifiercv'].alpha_

rcv_pipe_rest_pred = rcv_pipe_rest.predict(X = dfV2[dfV2.columns.drop(['note', 'rest', 'octave', 'duration'])])
plt.scatter(dfV2.rest, rcv_pipe_rest_pred, alpha = 0.01)

rcv_pipe_octave = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_octave.fit(X = dfV2[dfV2.rest != 1][dfV2.columns.drop(['note', 'octave', 'duration'])], 
                    y = dfV2[dfV2.rest != 1]['octave'])
rcv_pipe_octave['ridgeclassifiercv'].alpha_

rcv_pipe_octave_pred = rcv_pipe_octave.predict(X = dfV2[dfV2.rest != 1][dfV2.columns.drop(['note', 'octave', 'duration'])])
plt.scatter(dfV2[dfV2.rest != 1]['octave'], rcv_pipe_octave_pred, alpha = 0.01)

rcv_pipe_note = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'note' and i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_note.fit(X = dfV2[dfV2.rest != 1][dfV2.columns.drop(['note', 'duration'])], 
                  y = dfV2[dfV2.rest != 1]['note'])
rcv_pipe_note['ridgeclassifiercv'].alpha_

rcv_pipe_note_pred = rcv_pipe_note.predict(X = dfV2[dfV2.rest != 1][dfV2.columns.drop(['note', 'duration'])])
plt.scatter(dfV2[dfV2.rest != 1]['note'], rcv_pipe_note_pred, alpha = 0.01)

rcv_pipe_duration = make_pipeline(
    make_column_transformer(
        (OneHotEncoder(handle_unknown = 'ignore'), [i for i in cate_cols if (i != 'duration')])
    ),
    RidgeClassifierCV(alphas = [0.001, 0.01, 0.1, 1., 10., 100., 1000.], cv = 10)
)
## here we need to filter out rests
rcv_pipe_duration.fit(X = dfV2[dfV2.rest != 1][dfV2.columns.drop(['duration'])], 
                    y = dfV2[dfV2.rest != 1]['duration'])
rcv_pipe_note['ridgeclassifiercv'].alpha_

rcv_pipe_duration_pred = rcv_pipe_duration.predict(X = dfV2[dfV2.rest != 1][dfV2.columns.drop(['duration'])])
plt.scatter(dfV2[dfV2.rest != 1]['duration'], rcv_pipe_duration_pred, alpha = 0.01)
# this looks better.

# stochastic
dfV2_pred = dfV2_full
random.seed(1614)
for i in range(100):
    # making the next row with blanks
    temp_next_row = pd.DataFrame([np.nan, np.nan, np.nan, np.nan]+dfV2_pred.tail(1).iloc[:,:-4].values.tolist()[0][:]).T
    temp_next_row.columns = dfV2_pred.columns
    dfV2_pred = dfV2_pred.append(temp_next_row, ignore_index=True)
    
    # predict the next rest (not actually predicting)
    temp_v1_rest = 0
    dfV2_pred.iloc[-1,2] = temp_v1_rest
    # predict the next octave
    temp_v1_octave = rcv_pipe_octave.predict(X=dfV2_pred.tail(1).drop(columns = ['octave', 'note', 'duration']))
    dfV2_pred.iloc[-1,1] = temp_v1_octave
    # predict the next note
    # temp_v1_note = rcv_pipe_note.predict(X=dfV2_pred.tail(1).drop(columns = ['note', 'duration']))
    d = rcv_pipe_note.decision_function(X=dfV2_pred.tail(1).drop(columns = ['note', 'duration']))
    probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
    temp_v1_note = random.choices(rcv_pipe_note.classes_, weights=probs[0])[0]
    dfV2_pred.iloc[-1,0] = temp_v1_note
    # predict the next duration
    # temp_v1_duration = rcv_pipe_duration.predict(X=dfV2_pred.tail(1).drop(columns = ['duration']))
    d = rcv_pipe_duration.decision_function(X=dfV2_pred.tail(1).drop(columns = ['duration']))
    probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
    temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
    while(temp_v1_duration > 12):
        temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
    dfV2_pred.iloc[-1,3] = temp_v1_duration


# # %% [markdown]
# # ## Combining all voices

# # %%
# from midiutil import MIDIFile

# mymidi = MIDIFile(numTracks=4)
# mymidi.addTempo(track = 0, time = 0, tempo = 100)
# current_time = 0
# for i in range(len(dfV0_pred)):
#     current_note = int(dfV0_pred.iloc[i,0] + dfV0_pred.iloc[i,1]*12)
#     if(dfV0_pred.iloc[i,2] == 1):
#         current_time = current_time + int(dfV0_pred.iloc[i,3])
#         continue
#     mymidi.addNote(track = 0, channel = 0, pitch = current_note, time = current_time*0.25, duration = int(dfV0_pred.iloc[i,3])*0.25, volume = 100)
#     current_time = current_time + int(dfV0_pred.iloc[i,3])
    
# mymidi.addTempo(track = 1, time = 0, tempo = 100)
# current_time = 0
# for i in range(len(dfV1_pred)):
#     current_note = int(dfV1_pred.iloc[i,0] + dfV1_pred.iloc[i,1]*12)
#     if(dfV1_pred.iloc[i,2] == 1):
#         current_time = current_time + int(dfV1_pred.iloc[i,3])
#         continue
#     mymidi.addNote(track = 1, channel = 0, pitch = current_note, time = current_time*0.25, duration = int(dfV1_pred.iloc[i,3])*0.25, volume = 100)
#     current_time = current_time + int(dfV1_pred.iloc[i,3])
    
# mymidi.addTempo(track = 2, time = 0, tempo = 100)
# current_time = 0
# for i in range(len(dfV2_pred)):
#     current_note = int(dfV2_pred.iloc[i,0] + dfV2_pred.iloc[i,1]*12)
#     if(dfV2_pred.iloc[i,2] == 1):
#         current_time = current_time + int(dfV2_pred.iloc[i,3])
#         continue
#     mymidi.addNote(track = 2, channel = 0, pitch = current_note, time = current_time*0.25, duration = int(dfV2_pred.iloc[i,3])*0.25, volume = 100)
#     current_time = current_time + int(dfV2_pred.iloc[i,3])

# mymidi.addTempo(track = 3, time = 0, tempo = 100)
# current_time = 0
# for i in range(len(dfV3_pred)):
#     current_note = int(dfV3_pred.iloc[i,0] + dfV3_pred.iloc[i,1]*12)
#     if(dfV3_pred.iloc[i,2] == 1):
#         current_time = current_time + int(dfV3_pred.iloc[i,3])
#         continue
#     mymidi.addNote(track = 3, channel = 0, pitch = current_note, time = current_time*0.25, duration = int(dfV3_pred.iloc[i,3])*0.25, volume = 100)
#     current_time = current_time + int(dfV3_pred.iloc[i,3])
        
# with open("pred_all_voices.mid", "wb") as output_file:
#     mymidi.writeFile(output_file)

# # %%
# from midiutil import MIDIFile

# mymidi = MIDIFile(numTracks=4)
# mymidi.addTempo(track = 0, time = 0, tempo = 100)
# current_time = 0
# for i in range(len(dfV0_full)):
#     current_note = int(dfV0_full.iloc[i,0] + dfV0_full.iloc[i,1]*12)
#     if(dfV0_full.iloc[i,2] == 1):
#         current_time = current_time + int(dfV0_full.iloc[i,3])
#         continue
#     mymidi.addNote(track = 0, channel = 0, pitch = current_note, time = current_time*0.25, duration = int(dfV0_full.iloc[i,3])*0.25, volume = 100)
#     current_time = current_time + int(dfV0_full.iloc[i,3])
    
# mymidi.addTempo(track = 1, time = 0, tempo = 100)
# current_time = 0
# for i in range(len(dfV1_full)):
#     current_note = int(dfV1_full.iloc[i,0] + dfV1_full.iloc[i,1]*12)
#     if(dfV1_full.iloc[i,2] == 1):
#         current_time = current_time + int(dfV1_full.iloc[i,3])
#         continue
#     mymidi.addNote(track = 1, channel = 0, pitch = current_note, time = current_time*0.25, duration = int(dfV1_full.iloc[i,3])*0.25, volume = 100)
#     current_time = current_time + int(dfV1_full.iloc[i,3])
    
# mymidi.addTempo(track = 2, time = 0, tempo = 100)
# current_time = 0
# for i in range(len(dfV2_full)):
#     current_note = int(dfV2_full.iloc[i,0] + dfV2_full.iloc[i,1]*12)
#     if(dfV2_full.iloc[i,2] == 1):
#         current_time = current_time + int(dfV2_full.iloc[i,3])
#         continue
#     mymidi.addNote(track = 2, channel = 0, pitch = current_note, time = current_time*0.25, duration = int(dfV2_full.iloc[i,3])*0.25, volume = 100)
#     current_time = current_time + int(dfV2_full.iloc[i,3])

# mymidi.addTempo(track = 3, time = 0, tempo = 100)
# current_time = 0
# for i in range(len(dfV3_full)):
#     current_note = int(dfV3_full.iloc[i,0] + dfV3_full.iloc[i,1]*12)
#     if(dfV3_full.iloc[i,2] == 1):
#         current_time = current_time + int(dfV3_full.iloc[i,3])
#         continue
#     mymidi.addNote(track = 3, channel = 0, pitch = current_note, time = current_time*0.25, duration = int(dfV3_full.iloc[i,3])*0.25, volume = 100)
#     current_time = current_time + int(dfV3_full.iloc[i,3])
        
# with open("original_all_voices.mid", "wb") as output_file:
#     mymidi.writeFile(output_file)

# %% Performance

v_0 = dfV0_pred[['note', 'octave', 'rest', 'duration']]
v_1 = dfV1_pred[['note', 'octave', 'rest', 'duration']]
v_2 = dfV2_pred[['note', 'octave', 'rest', 'duration']]
v_3 = dfV3_pred[['note', 'octave', 'rest', 'duration']]

# np.savetxt("m_all_0.txt",v_0)
# np.savetxt("m_all_1.txt",v_1)
# np.savetxt("m_all_2.txt",v_2)
# np.savetxt("m_all_3.txt",v_3)

np.savetxt("m_pre_0.txt",v_0)
np.savetxt("m_pre_1.txt",v_1)
np.savetxt("m_pre_2.txt",v_2)
np.savetxt("m_pre_3.txt",v_3)



