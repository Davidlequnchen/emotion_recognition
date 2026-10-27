import parselmouth
import os.path
import numpy as nupmy
import pandas as pd
from parselmouth.praat import call
import librosa
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler

def calculateF0Min(voiceID):
    # Load audio file
    y, sr = librosa.load(voiceID)
    # Calculate F0 using librosa.pyin
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    # Filter out unvoiced values
    f0_voiced = f0[voiced_flag]
    # Calculate minimum and maximum F0
    minF0 = nupmy.min(f0_voiced)
    return minF0
def calculateF0Max(voiceID):
    y, sr = librosa.load(voiceID)
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
    f0_voiced = f0[voiced_flag]
    maxF0 = nupmy.max(f0_voiced)
    return maxF0

def measureIntensity(voiceID):
    sound = parselmouth.Sound(voiceID)
    intensity = call(sound, "To Intensity", 75,0.025, 500)
    meanIntensity = call(intensity, "Get mean", 0, 0)
    return meanIntensity

def measureRMSEnergy(voiceID):
    sound = parselmouth.Sound(voiceID)
    rmsEnergy = nupmy.sqrt(nupmy.mean(sound.values**2))
    return rmsEnergy

def measureSpectralCOG(voiceID, sr=44100):
    # Load audio file
    y, sr = librosa.load(voiceID, sr=sr)
    # Calculate the spectral centroid
    spectralCOG = librosa.feature.spectral_centroid(y=y, sr=sr)[0] #,n_fft=1024,hop_length=512
    spectralCOG = nupmy.mean(spectralCOG)
    return spectralCOG
def measureSpectralSpread(voiceID, sr=44100):
    # Load audio file
    y, sr = librosa.load(voiceID, sr=sr)
    # Calculate the spectral spread
    spectralSpread = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    # Calculate the mean spectral spread
    spectralSpread = nupmy.mean(spectralSpread)
    return spectralSpread
def measurePitch(voiceID, f0min, f0max, unit):
        sound = parselmouth.Sound(voiceID)  # read the sound
        duration = call(sound, "Get total duration")  # duration
        pitch = call(sound, "To Pitch", 0.0, f0min, f0max)  # create a praat pitch object
        meanF0 = call(pitch, "Get mean", 0, 0, unit)  # get mean pitch
        stdevF0 = call(pitch, "Get standard deviation", 0, 0, unit)  # get standard deviation
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
        localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        meanIntensity = measureIntensity(voiceID)
        rmsEnergy = measureRMSEnergy(voiceID)
        spectralCOG = measureSpectralCOG(voiceID)
        spectralSpread = measureSpectralSpread(voiceID)
        return duration, meanF0, stdevF0, hnr, localJitter, localShimmer,meanIntensity, rmsEnergy, spectralCOG, spectralSpread

# append audio file paths
folder_path = "/Users/mengyuanli/Documents/..." #replace the corpus file path, total audio number is 7579
file_list = [file for file in os.listdir(folder_path) if file.endswith(".wav")]
print(file_list)
results=[]
df_result1 = pd.DataFrame()

# extract "duration", "meanF0", "stdevF0", "hnr", "localJitter", "localShimmer","meanIntensity", "rmsEnergy", "spectralCOG", "spectralSpread"
for filename in file_list:
    voice_path = os.path.join(folder_path, filename)
    result = measurePitch(voice_path, 75, 450, "Hertz")
    results.append((filename,) + result)
    # print(results)
print(results)

columns = ["filename","duration", "meanF0", "stdevF0", "hnr", "localJitter", "localShimmer",
           "meanIntensity", "rmsEnergy", "spectralCOG", "spectralSpread"]
df_result1 = pd.DataFrame(results, columns=columns)
print(df_result1)
df_result1.to_csv("acoustic information1.csv")

result2=[]
#extract 'max_f0' and 'min_f0'
folder_path = "/Users/mengyuanli/Documents/..."#replace the corpus file path, total audio number is 7579
file_list = [file for file in os.listdir(folder_path) if file.endswith(".wav")]
for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    snd = parselmouth.Sound(file_path)
    pitch = snd.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values != 0]

    if len(f0_values) > 0:  # Check if the array is not empty
        f0_values = f0_values[(f0_values != 0)]  # Remove leading/trailing zeros
        max_f0 = nupmy.max(f0_values)
        min_f0 = nupmy.min(f0_values)
        result2.append((filename, max_f0, min_f0))
    else:
        result2.append((filename, nupmy.nan, nupmy.nan))  # Handle case where array is empty

df_result2 = pd.DataFrame(result2, columns=['filename', 'max_f0', 'min_f0'])
print(df_result2)
# df_result2.to_csv("acoustic information2.csv")

df=pd.merge(df_result1, df_result2, left_on = "filename", right_on="filename", how="right")
df.to_excel("acoustic information.xlsx")

#match acoustic information for for following analysis
data=pd.read_excel("audio information.xlsx")
df['filename'] =df['filename'].apply(lambda x: x[0:5])
print(df['filename'])
data['filename'] = data['filename'].astype(str)
df=pd.merge(df, data, left_on = "filename", right_on="filename", how="right")
print(df)

#results of table 6
result = df.groupby(['emotioncode']).agg({'duration': ['mean'], 'meanF0': ['mean'], 'stdevF0': ['mean']
                                                   , 'hnr': ['mean'], 'localJitter': ['mean'], 'localShimmer': ['mean']
                                                   , 'meanIntensity': ['mean'], 'rmsEnergy': ['mean'], 'spectralCOG': ['mean']
                                                   , 'spectralSpread': ['mean'], 'max_f0': ['mean'], 'min_f0': ['mean']})

result.reset_index(inplace=True)
print(result)#this is the results of table 6
# result.to_excel("acoustic analysis-table 6.xlsx")

#VIF collinearity test
df['Intercept'] = 1

columns=['duration', 'meanF0', 'min_f0', 'max_f0','stdevF0', 'hnr', 'localJitter', 'localShimmer'
                     , 'meanIntensity', 'rmsEnergy', 'spectralCOG', 'spectralSpread']
correlation_matrix = df[columns].corr()
print("Correlation Matrix:")
print(correlation_matrix)

variables = df[['duration','meanF0', 'min_f0', 'max_f0','stdevF0', 'hnr', 'localJitter', 'localShimmer'
             ,  'rmsEnergy', 'spectralCOG'] ]
vif_data = pd.DataFrame()
vif_data["Variable"] = variables.columns
vif_data["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]

print("\nVIF:")
print(vif_data)

#regression analysis
categories = df['emotioncode'].unique()

for category in categories:
    df_category = df[df['emotioncode'] == category]
    y = df_category['correct_1']
    X = df_category[['duration','meanF0', 'min_f0', 'max_f0','stdevF0', 'hnr', 'localJitter', 'localShimmer'
                     , 'meanIntensity', 'rmsEnergy', 'spectralCOG', 'spectralSpread']]

    X = sm.add_constant(X)
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)

    model = sm.OLS(y, X).fit()
    model2= sm.OLS(y, sm.add_constant(X_standardized)).fit()

    print(f"Regression results for Category {category}:")
    print(model.summary())
    print("\n")
    print(model2.summary())
    print("\n")





