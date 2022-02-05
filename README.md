# MLProject2021

Training data: 

- `F.txt` The training data for Bach's fugue.
- `twinkle.txt`: *Twinkle Twinkle Little Star*.

Outputs from `single_voice_prediction.ipynb`:

- `original.mid`: The original data (F.txt), only the lowest voice.
- `pred.mid`: The original (F.txt) plus its predictions, only the lowest voice. (You may compare this with `original.mid` to see when the prediction starts.)
- `twinkle_pred.mid`: The twinkle.txt data plus its predictions, only the 0th (first column).
- `original_all_voices.mid`: The original data (F.txt), all voices. *Not used in the report.*
- `pred_all_voices.mid`: The original (F.txt) plus its predictions, all voices. These voices were predicted separately. *Not used in the report.*

Outputs from `multiple_voice_prediction.ipynb`:

- `simplified_generated.mid`: The multiple-voice prediction.

Performance results:
- `p4Chain.py`: add split methods for test and train set, and generate both regression outputs for all data used and only train data used
- `p4multi.py`: the same as above
- `Performance.rmd`: calculate piano format output for both algorithms and make plots
- `statistics_calculation.py`: calculate statistics parameteres
