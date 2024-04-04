import pandas as pd
import numpy as np
from tqdm import tqdm

NUMBER_OF_EXPERIMENTS = 7
columns_to_look = ['EEG.Fp1',	'EEG.AF3',	'EEG.F3',	'EEG.FC1'	,'EEG.C3'	,'EEG.FC3'	,'EEG.T7'	,'EEG.CP5',	'EEG.CP1'	,'EEG.P1'	,'EEG.P7'	,'EEG.P9',	'EEG.PO3'	,'EEG.O1'	,'EEG.O9',	'EEG.POz',	'EEG.Oz',	'EEG.O10',	'EEG.O2',	'EEG.PO4'	,'EEG.P10',	'EEG.P8',	'EEG.P2','EEG.CP2',	'EEG.CP6'	,'EEG.T8',	'EEG.FC4',	'EEG.C4',	'EEG.FC2',	'EEG.F4', 'EEG.AF4',	'EEG.Fp2']
TRIM_FIRST = 50
TRIM_LAST = 50

all_dfs = []
for i in range(1, NUMBER_OF_EXPERIMENTS + 1):
    print("Running for Exp",i )
    data = pd.read_csv("data/exp{}/data.csv".format(i), skiprows=1)
    column_names_extra = [col for col in data.columns if col.startswith("POW")]

    markers = pd.read_csv("data/exp{}/intervalMarker.csv".format(i))
    markers = markers[['type','duration','marker_id']]
    # find how is the marker called for open/closed eyes
    eyes_close_marker = markers[markers['type'].str.startswith('eyesclose')]['type'].values[0]
    eyes_open_marker = markers[markers['type'].str.startswith('eyesopen')]['type'].values[0]
    # take the important markers only
    markers = markers[markers['type'].isin( [eyes_close_marker, eyes_open_marker,
                                          '0','1','2','3','4', '5','6', '7','8', '9'])]
    # we are using closed eyes for baseline
    baseline_marker_closed_eyes = markers[markers['type'] == eyes_close_marker]
    # find where baseline starts and ends bsaed on the marker ID
    index1 = data[data['MarkerIndex'] == baseline_marker_closed_eyes['marker_id'].values[0]].index[0]
    index2 = data[data['MarkerIndex'] == -baseline_marker_closed_eyes['marker_id'].values[0]].index[0]
    # slice the data based on found indexes
    baseline_df = data.iloc[index1:index2+1]
    filtered_df_channels = baseline_df[columns_to_look]
    # compute mean accross all channels which represents baseline
    baseline_closed = np.mean(filtered_df_channels, axis=0)
    baseline_closed = baseline_closed[columns_to_look]
    # this is the same procedure for open eyes baseline which are not using (probably0
    baseline_marker_open_eyes = markers[markers['type'] == eyes_open_marker]
    index1 = data[data['MarkerIndex'] == baseline_marker_closed_eyes['marker_id'].values[0]].index[0]
    index2 = data[data['MarkerIndex'] == -baseline_marker_closed_eyes['marker_id'].values[0]].index[0]
    baseline_df = data.iloc[index1:index2+1]
    filtered_df_channels = baseline_df[columns_to_look]
    baseline_open = np.mean(filtered_df_channels, axis=0)
    baseline_open = baseline_open[columns_to_look]

    # NOW WE ARE NORMALIZING MEASURMENTS AND PRODUCING FILES

    # we sort markers by type
    for index,mark in tqdm(markers.sort_values('type', ascending=False).iterrows(),total=len(markers),
                           desc="Processing rows"):
        # we skip first image and we skip the baseline measurments
        if mark['marker_id'] != 10: # first 2 is ID 10
            # slice the data for certain number
            index1 = data[data['MarkerIndex'] == mark['marker_id']].index[0]
            index2 = data[data['MarkerIndex'] == -mark['marker_id']].index[0]
            filtered_df = data.iloc[index1:index2+1]
            # check the duration
            duration = data.loc[index2,'Timestamp'] - data.loc[index1,'Timestamp']
            # compare duration with marker duration. This should be roughly the same
            assert round(duration, 0) == round(mark['duration'], 0)
            # take only columns of interest
            filtered_df_channels = filtered_df[columns_to_look]
            # remove first and last data in order to remove noise
            filtered_df_channels = filtered_df_channels.iloc[TRIM_FIRST:-TRIM_LAST]
            # take extra columns (alpha beta, gama waves)
            filtered_df_alpha_beta = filtered_df[column_names_extra].dropna()
            filtered_df_alpha_beta.to_csv('data/processed/alphaBeta/{}_{}_exp{}.csv'.
                                        format(mark['type'], mark['marker_id'], i))
            # save the file which is not normalized into "PROCESSED"
            filtered_df_channels.to_csv('data/processed/channels/{}_{}_exp{}.csv'.
                                        format(mark['type'], mark['marker_id'], i))
            # add number which is observed (stimuli) add exp number and deduct baseline measurment
            filtered_df_channels['label'] = mark['type']
            filtered_df_channels_baselined = filtered_df_channels-baseline_closed.T
            filtered_df_channels_baselined['label'] = mark['type']
            filtered_df_channels_baselined['exp'] = i
            # baselines go to another folder
            if mark['type'] == eyes_close_marker or mark['type'] == eyes_open_marker:
                filtered_df_channels_baselined.to_csv('data/normalized_by_baseline/for_normalization/{}_{}_exp{}.csv'.
                                            format(mark['type'], mark['marker_id'], i))
            else:
                filtered_df_channels_baselined.to_csv('data/normalized_by_baseline/channels/{}_{}_exp{}.csv'.
                                            format(mark['type'], mark['marker_id'], i))

            all_dfs.append(filtered_df_channels_baselined)

total_data = pd.concat(all_dfs, ignore_index=True)



