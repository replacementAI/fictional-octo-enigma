#pip install praat-parselmouth

import pandas as pd
import parselmouth
from parselmouth.praat import call

def extract_audio_features(sound):
    pitch = sound.to_pitch()
    
    f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    
    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    jitter_local = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_abs = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rap = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddp = rap * 3

    pointProcess = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    
    shimmer_local = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_db = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3 = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq5 = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    dda = apq3 * 3
    
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    return {
        "MDVP:Fo(Hz)": f0, "MDVP:Fhi(Hz)": fhi, "MDVP:Flo(Hz)": flo,
        "MDVP:Jitter(%)": jitter_local, "MDVP:Jitter(Abs)": jitter_abs, 
        "MDVP:RAP": rap, "MDVP:PPQ": ppq, "Jitter:DDP": ddp,
        "MDVP:Shimmer": shimmer_local, "MDVP:Shimmer(dB)": shimmer_db,
        "Shimmer:APQ3": apq3, "Shimmer:APQ5": apq5, "Shimmer:DDA": dda, "HNR": hnr
    }

path = "/content/parkinsons.zip"
import zipfile
with zipfile.ZipFile(path, 'r') as zip_ref:
    zip_ref.extractall('/content/parkinsons')

file_path = '/content/parkinsons/parkinsons.data'
raw = pd.read_csv(file_path)
columns_to_drop = ['D2','DFA','spread1','spread2','PPE','RPDE','NHR','MDVP:APQ']
y = raw['status']
X = raw.drop(columns=['name', 'status']+columns_to_drop)

input_audio = parselmouth.Sound("/content/input_audio.mp3")
best_params = {'learning_rate': 0.05, 'max_iter': 20, 'max_leaf_nodes': 3}
best_model = HistGradientBoostingClassifier(**best_params)
best_model.fit(X, y)
to_pred = pd.DataFrame([extract_audio_features(input_audio)])
predictions = best_model.predict(to_pred)
