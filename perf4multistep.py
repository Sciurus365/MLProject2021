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
num_predict = 100

# Import the csv
piano_input = pd.read_csv('F.txt', sep='\t', header=None)

# Transform piano key pitch (A0 = 1) to Midi pitch (A0 = 21)
# Different sources say different numbers for A0. Might be 9 or 21. :| 9 seems to be A(-1) generally though.
# If we map octave -1 to index 0 (which is the easiest probably), then we need A0 = 21.
midi_input = piano_input.where(piano_input == 0, piano_input + 20)
# Take the voice we are working on
voice_input = midi_input.iloc[:, VOICE]

# Split train and test data set
voice_test = voice_input.iloc[-num_predict:]
voice_input = voice_input.drop(voice_input.tail(num_predict).index)

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

# Split test and train data set
test_notes = notes_df.tail(num_predict)
notes_df = notes_df.drop(notes_df.tail(num_predict).index)

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

# %% [markdown]
# # Models

# %%
dfV1_full

# %% [markdown]
# Change here: duration also as categorical

# %%
dfV1 = notes_lagged_df
# columns as categories or contineous variables
cate_cols = [i for i in dfV1.columns.values if (re.search('note', i) or re.search('rest', i) or re.search('duration', i))]
cont_cols = [i for i in dfV1.columns.values if (re.search('octave', i))]
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
rcv_pipe_rest.fit(X = dfV1[dfV1.columns.drop(['rest', 'note', 'octave', 'duration'])], y = dfV1['rest'])
rcv_pipe_rest['ridgeclassifiercv'].alpha_

# %%
rcv_pipe_rest_pred = rcv_pipe_rest.predict(X = dfV1[dfV1.columns.drop(['note', 'rest', 'octave', 'duration'])])
plt.scatter(dfV1.rest, rcv_pipe_rest_pred, alpha = 0.01)
# it never predicts a rest, which is fine?
# then we don't need to predict duration for rests.

# %% [markdown]
# ## model for `octave`

# %%
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

# %%
rcv_pipe_octave_pred = rcv_pipe_octave.predict(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['note', 'octave', 'duration'])])
plt.scatter(dfV1[dfV1.rest != 1]['octave'], rcv_pipe_octave_pred, alpha = 0.01)

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
rcv_pipe_note.fit(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['note', 'duration'])], 
                  y = dfV1[dfV1.rest != 1]['note'])
rcv_pipe_note['ridgeclassifiercv'].alpha_

# %%
rcv_pipe_note_pred = rcv_pipe_note.predict(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['note', 'duration'])])
plt.scatter(dfV1[dfV1.rest != 1]['note'], rcv_pipe_note_pred, alpha = 0.01)

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
rcv_pipe_duration.fit(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['duration'])], 
                    y = dfV1[dfV1.rest != 1]['duration'])
rcv_pipe_note['ridgeclassifiercv'].alpha_

# %%
rcv_pipe_duration_pred = rcv_pipe_duration.predict(X = dfV1[dfV1.rest != 1][dfV1.columns.drop(['duration'])])
plt.scatter(dfV1[dfV1.rest != 1]['duration'], rcv_pipe_duration_pred, alpha = 0.01)
# this looks better.

# %% [markdown]
# # Prediction

# %%
# stochastic
dfV1_pred = dfV1_full
random.seed(1614)
for i in range(num_predict):
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

# %%
# Test performance
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

y_test = voice_test
y_pred = dfV1_pred.tail(num_predict)['note']

# Plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1,figsize=(7,7),dpi=300)
ax.set_title('Predition VS Test')
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--',c='k', label="1:1 line")
plt.xlabel('Test data')
plt.ylabel('Predicted data')
ax.scatter(y_test,y_pred,c = 'b',marker = 'o' , alpha = 0.1)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--',c='k', label="1:1 line")
plt.show()

# Parameters for regression
MSE = mean_squared_error(y_test,y_pred)
print("Following are for Bach")
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_pred)
print("R2 = ", R2)

# Parameters for multilabel classification
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# y_true = test_notes
# y_pred = dfV1_pred.tail(num_predict)[['note', 'octave', 'rest', 'duration']]

# # accuracy_score
# acc = accuracy_score(y_true, y_pred)



# %% [markdown]
# ## The same procedure, but for Little Star

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
num_predict = 10

# Import the csv
piano_input = pd.read_csv('twinkle.txt', sep='\t', header=None)

# Transform piano key pitch (A0 = 1) to Midi pitch (A0 = 21)
# Different sources say different numbers for A0. Might be 9 or 21. :| 9 seems to be A(-1) generally though.
# If we map octave -1 to index 0 (which is the easiest probably), then we need A0 = 21.
midi_input = piano_input.where(piano_input == 0, piano_input - 12)
# Take the voice we are working on
voice_input = midi_input.iloc[:, VOICE]

# Split train and test data set
# voice_test = voice_input.iloc[-num_predict:]
# voice_input = voice_input.drop(voice_input.tail(num_predict).index)

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

# Split test and train data set
test_notes = notes_df.tail(num_predict)
notes_df = notes_df.drop(notes_df.tail(num_predict).index)

# Create a lagged input dataframe
notes_lagged_df = notes_df.copy()

for lag in range(1, LAG + 1):
    lagged = notes_df.shift(lag)
    lagged.columns = [f'{col_name}_lag{lag}' for col_name in notes_df.columns]
    notes_lagged_df = pd.concat((notes_lagged_df, lagged), axis=1)

# Drop rows containing NA (i.e. the first LAG rows basically)
dfV1_twinkle_full = notes_lagged_df
notes_lagged_df = notes_lagged_df.dropna()

# Show first 10 notes
notes_lagged_df.head(10)

# %%
dfV1_twinkle_full.head(10)

# %%
# stochastic
dfV1_twinkle = dfV1_twinkle_full
random.seed(1614)
for i in range(num_predict):
    # making the next row with blanks
    temp_next_row = pd.DataFrame([np.nan, np.nan, np.nan, np.nan]+dfV1_twinkle.tail(1).iloc[:,:-4].values.tolist()[0][:]).T
    temp_next_row.columns = dfV1_twinkle.columns
    dfV1_twinkle = dfV1_twinkle.append(temp_next_row, ignore_index=True)
    
    # predict the next rest (not actually predicting)
    temp_v1_rest = 0
    dfV1_twinkle.iloc[-1,2] = temp_v1_rest
    # predict the next octave
    temp_v1_octave = rcv_pipe_octave.predict(X=dfV1_twinkle.tail(1).drop(columns = ['octave', 'note', 'duration']))
    dfV1_twinkle.iloc[-1,1] = temp_v1_octave
    # predict the next note
    # temp_v1_note = rcv_pipe_note.predict(X=dfV1_twinkle.tail(1).drop(columns = ['note', 'duration']))
    d = rcv_pipe_note.decision_function(X=dfV1_twinkle.tail(1).drop(columns = ['note', 'duration']))
    probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
    temp_v1_note = random.choices(rcv_pipe_note.classes_, weights=probs[0])[0]
    dfV1_twinkle.iloc[-1,0] = temp_v1_note
    # predict the next duration
    # temp_v1_duration = rcv_pipe_duration.predict(X=dfV1_twinkle.tail(1).drop(columns = ['duration']))
    d = 3.5*rcv_pipe_duration.decision_function(X=dfV1_twinkle.tail(1).drop(columns = ['duration']))
    probs = [np.exp(i) / np.sum(np.exp(i)) for i in d]
    temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
    while(temp_v1_duration > 12):
        temp_v1_duration = random.choices(rcv_pipe_duration.classes_, weights=probs[0])[0]
    dfV1_twinkle.iloc[-1,3] = temp_v1_duration

# %%
# Test performance
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

y_test = test_notes['note']
y_pred = dfV1_twinkle.tail(num_predict)['note']

# Plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1,figsize=(7,7),dpi=300)
ax.set_title('Predition VS Test in Twinkle Stars')
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--',c='k', label="1:1 line")
plt.xlabel('Test data')
plt.ylabel('Predicted data')
ax.scatter(y_test,y_pred,c = 'b',marker = 'o' , alpha = 0.1)
ax.plot((0, 1), (0, 1), transform=ax.transAxes, ls='--',c='k', label="1:1 line")
plt.show()

# Parameters for regression
print("Following are for stars")
MSE = mean_squared_error(y_test,y_pred)
print("MSE = " , MSE)
MAE = mean_absolute_error(y_test,y_pred)
print("MAE = ", MAE)
R2 = r2_score(y_test,y_pred)
print("R2 = ", R2)

# # Parameters for multilabel classification
# y_true = test_notes
# y_pred = dfV1_twinkle.tail(num_predict)[['note', 'octave', 'rest', 'duration']]

# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score

# m = MultiLabelBinarizer().fit(y_true)

# f1 = f1_score(m.transform(y_true),
#           m.transform(y_pred),
#           average='macro')
# print("F1 = ",f1)

# acc = accuracy_score(m.transform(y_true), m.transform(y_pred))
# print("Accuracy score is ", acc)



