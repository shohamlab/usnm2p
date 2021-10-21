# -*- coding: utf-8 -*-
# @Author: Theo Lemaire
# @Date:   2021-10-12 16:46:18
# @Last Modified by:   Theo Lemaire
# @Last Modified time: 2021-10-19 14:18:50

try:
    res_list = process_map(get_file_table, s2p_dirs, mouse_list, region_list, mpa_list, dc_list)
except ValueError: # this happens if there is a mismatch in shapes. If so, try with a fix for deepinterpolation.
    clear_output()
    res_list = process_map(get_file_table, s2p_dirs, mouse_list, region_list, mpa_list, dc_list, [True] * len(dc_list)) # sets fix convolution to true
    
    
res_tab = pd.concat([x[0] for x in res_list])
trial_pd = pd.concat([x[1] for x in res_list])

print('res_tab:')
display(res_tab)

print('trial_pd')
display(trial_pd)

cellstats = res_tab.groupby(['mpa','dc','resp_type','mouse_reg','roi'], as_index=False, observed=True).agg({'dff_resp':['mean', 'std','count']})
cellstats.columns = ['mpa', 'dc', 'resp_type', 'mouse_reg', 'roi', 'dff_resp', 'dff_std', 'nsamples']
cellstats['dff_stderr'] = cellstats.dff_std / np.sqrt(cellstats.nsamples)
print('cellstats')
display(cellstats)

# do some cleaning
# TODO change naming inside the main function instead of this spaguetti code approach


def clean_dataframe(df):
    df = df.rename(columns=column_rename)
    for key, value in responder_rename.items():
        df.loc[df['Responder class'] == key, 'Responder class'] = value
    # remove outlier reponses
    df = df[(df['\u0394F/F'] < dff_outlier) & (df['\u0394F/F'] > -dff_outlier)]
    return df

trial_pd, cellstats, res_tab = map(clean_dataframe, [trial_pd, cellstats, res_tab])
    
responder_class_dict = cellstats.drop_duplicates(subset=['roi','mouse/region'], keep='first')['Responder class'].value_counts().to_dict()
responder_class_legend = ', '.join(f"{key}={to_unicode(val)}" for (key,val) in responder_rename.items())
trial_pd['Time (s)'] =  trial_pd.frame / fps - stim_frame / fps # time in seconds