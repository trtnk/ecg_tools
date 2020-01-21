import numpy as np
import pandas as pd
import neurokit as nk

feature_names = ["RMSSD",
                 "meanNN",
                 "sdNN",
                 "cvNN",
                 "CVSD",
                 "medianNN",
                 "madNN",
                 "mcvNN",
                 "pNN50",
                 "pNN20",
                 "Triang",
                 "Shannon_h",
                 "Triang",
                 "Shannon_h",
                 "ULF",
                 "VLF",
                 "LF",
                 "HF",
                 "VHF",
                 "Total_Power",
                 "LFn",
                 "HFn",
                 "LF/HF",
                 "LF/P",
                 "HF/P",
                 "DFA_1",
                 "DFA_2",
                 "Shannon",
                 "Sample_Entropy",
                 "Correlation_Dimension",
                 "Correlation_Dimension",
                 "Entropy_Multiscale_AUC",
                 "Entropy_SVD",
                 "Entropy_Spectral_VLF",
                 "Entropy_Spectral_LF",
                 "Entropy_Spectral_HF",
                 "Fisher_Info",
                 "FD_Petrosian",
                 "FD_Higushi"]

# ecg: Raw ECG signal
# sample_rate: sampling freqency
# extract_range: Time (sec) used for a feature extraction (default: all)
# shift_range: Time (sec) to shift window (default: same as extract_range)
# labels: label information about ecg (This must be the same length as ecg)
def extract_ecg_features(ecg, sample_rate, extract_range=None, shift_range=None, hrv_features=["time", "frequency", "nonlinear"], labels=None):
    if type(ecg) != np.ndarray:
        raise Exception("data type of ecg must be numpy.ndarray.")
    if ecg.ndim != 1:
        raise Exception("ecg dimention must be one.")
    if type(labels) == np.ndarray:
        if len(ecg) != len(labels):
            raise Exception("labels size must be the same as ecg")
        else:
            df_columns = ["index", "sample_rate", "range_sec"]+feature_names+["label"]
    else:
        df_columns = ["index", "sample_rate", "range_sec"]+feature_names
    if type(ecg) == list:
        ecg = np.array(ecg)
    if extract_range == None:
        extract_range = int((len(ecg))/sample_rate)
        print(extract_range)
    if shift_range == None:
        shift_range = extract_range

    current_ind = 0
    df_values = []
    print(current_ind+extract_range*sample_rate)
    while current_ind+extract_range*sample_rate <= len(ecg):
        values = [current_ind, sample_rate, extract_range]
        tmp_ecg = np.copy(ecg[current_ind:int(current_ind+extract_range*sample_rate)])
        processed_ecg_dict = nk.ecg_process(tmp_ecg, sampling_rate=sample_rate, quality_model=None, hrv_features=hrv_features)
        for feature_name in feature_names:
            values.append(processed_ecg_dict["ECG"]["HRV"].get(feature_name))
        if type(labels) == np.ndarray:
            cnt = np.bincount(labels[current_ind:int(current_ind+extract_range*sample_rate)])
            label = np.argmax(cnt)
            values.append(label)
        df_values.append(values)
        current_ind += shift_range*sample_rate
    feature_df = pd.DataFrame(np.array(df_values), columns=df_columns)
    return feature_df

# test
test_ecg = nk.ecg_simulate(duration=60*10) # create a dammy ecg (60sec)
labels = np.array([0]*int(len(test_ecg)/2)+[1]*int(len(test_ecg)/2))
out = extract_ecg_features(test_ecg, 1000, extract_range=60, labels=labels)