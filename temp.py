import os
import numpy as np

def get_file_table(mydir, mouse, region, mpa_list_single, dc_list_single, fix_conv=False):
    '''
    Using the suite2p output files, generate tables containing trial-by-trial or
    frame-by-frame normalized fluorescence info for each cell.

    :param mydir: root diectory???
    :param mouse: mouse name
    :param region: region name:
    :param 
    
    '''

    # print(mydir, mouse, region)

    iscell = np.load(os.path.join(mydir,'iscell.npy'), allow_pickle=True)[:,0] # the full "iscell" has a second column with the probability of being a cell
    cell_idx = np.array(iscell.nonzero()).reshape(-1)
    
    # limit ourselves to predicted cells
    stat = np.load(os.path.join(mydir,'stat.npy'), allow_pickle=True)[cell_idx]
    F = np.load(os.path.join(mydir,'F.npy'), allow_pickle=True)[cell_idx]
    Fneu = np.load(os.path.join(mydir,'Fneu.npy'), allow_pickle=True)[cell_idx]
    spks = np.load(os.path.join(mydir,'spks.npy'), allow_pickle=True)[cell_idx]
        
#     print('num cells: {} out of {} source ROIs'.format(len(F),len(iscell)))
#     print(F.shape)
        
    if fix_conv:
        F, Fneu, spks = (fix_start_end(data) for data in (F, Fneu, spks))
#         print(f'New shape, after fixing start and end: {F.shape}')

#     # estimation of neuropil coeficient for each cell, and subtraction
      # this did not work well
#     new_f = np.zeros_like(F)
#     for neuron, (trace, neuropil) in enumerate(zip(F, Fneu)):
#         linear_model = sm.RLM(trace, neuropil)
#         neuropil_factor = linear_model.fit().params[0]
#         # neuropil adjustment - consider doing a robust linear regression for F vs Fneu (including intercept) to calculate individual neuropil coeffs!
#         new_f[neuron] = trace - neuropil_factor * neuropil
        
#     F = new_f # reassign for downstream analysis compatibility

    # neuropil adjustment - consider doing a robust linear regression for F vs Fneu (including intercept) to calculate individual neuropil coeffs!
    F = F - neuropil_factor * Fneu

    # set pre-stim and post-stim windows here
    prestim = slice(stim_frame-7,stim_frame-1) # omit first second, not sure what's up with the data there. Hence start at 7 instead of 10
    poststim = slice(stim_frame+1, stim_frame+7) # assume peak response happens within 1 second of stimulation. Hence 3 first frames after stim
    
    # clean stimulus artifact
    if clean_artifact:
        dim1, dim2 = F.shape
        f_r = F.reshape((F.shape[0],len(mpa_list_single),-1,trial_length), order='C') # reshape data by trials and conditions
        best_guess = np.mean(f_r[:,:,:,prestim], axis=-1) # copy last data point
        f_r[:,:,:,stim_frame] = best_guess # replace artifact by pre-stim trial mean
        F = f_r.reshape((dim1,dim2)) # reassign F for compatibility with downstream analysis and undo reshape
            
    # reshape traces into (roi, mov, trial, frame) format
    # currently: ignore the first trial of each video due to the transition between recordings
    trial_reshaped_F = F.reshape((F.shape[0],len(mpa_list_single),-1,trial_length), order='C')[:,:,1:,:]
    trial_mean_F = F.reshape((F.shape[0],len(mpa_list_single),-1,trial_length), order='C')[:,:,1:,:].mean(axis=2)
    trial_mean_S = spks.reshape((spks.shape[0],len(mpa_list_single),-1,trial_length), order='C')[:,:,1:,:].mean(axis=2)
    
    # use z-score to determine class of response (positive, negative, non-responder/weak AKA undefined)
    # reshaped so compute_zscore() works
    f_timeseries = trial_reshaped_F.reshape((-1, trial_reshaped_F.shape[-2]*trial_reshaped_F.shape[-1]))
    p = compute_zscore(f_timeseries, axis=1) # compute z-score along the whole recording in each condition
    # undo reshape
    p = p.reshape((F.shape[0],len(mpa_list_single),-1,trial_length), order='C')
    p_max = p[:,:,:,poststim].mean(axis=2) # average across trials for each cell and condition
    p_min = p[:,:,:,poststim].mean(axis=2) # average across trials for each cell and condition
    
    # highest z-score in the poststim slice interval, per cell and condition. Then take max among all conditions for each cell
    p_max = np.max(p_max.max(axis=2), axis=1)
    # ibidem for min but with minimums
    p_min = np.min(p_min.min(axis=2), axis=1)
    # We are left with a (cell x z-score) vector
    
    responders = (p_max >= high_cutoff).nonzero()[0]
    neg_responders = ((p_max < high_cutoff) & (p_min <= low_cutoff)).nonzero()[0]

    # calculate df/f from pre- and post-stim windows
    dff_change = F.reshape((F.shape[0],len(mpa_list_single),-1,trial_length),order='C')[:,:,1:,:]
    dff_change = (dff_change[:,:,:,poststim].mean(axis=-1) - dff_change[:,:,:,prestim].mean(axis=-1)) / dff_change[:,:,:,prestim].mean(axis=-1)
#     F_reshape = F.reshape((F.shape[0],len(mpa_list_single), -1),order='C')[:,:]
#     baseline = np.tile(np.expand_dims(np.median(F_reshape, axis=2), axis=2), F_reshape.shape[2])
#     dff = (F_reshape - baseline) / baseline
#     dff = dff.reshape((dff.shape[0],len(mpa_list_single),-1,trial_length),order='C')[:,:,1:,:]
#     dff_change = dff[:,:,:,poststim].max(axis=3)
    
    # calculate spiking change from pre- and post-stim windows
    # currently: normalize by the average spike rate over that recording
    spk_change = spks.reshape((spks.shape[0],len(mpa_list_single),-1,trial_length),order='C')[:,:,1:,:]
    spk_change = (spk_change[:,:,:,poststim].mean(axis=-1) - spk_change[:,:,:,prestim].mean(axis=-1)) / spk_change.mean(axis=(2,3), keepdims=True).reshape(spk_change.shape[:2] + (1,))
    spk_change_mean = dff_change.mean(axis=(1,2))
    
    # transform dff change matrix and associated cell/mov/trial indices into table-based column format
    ind_array = np.indices(dff_change.shape) # cell_inds, mov_inds, trial_inds
    cell_inds, mov_inds, trial_inds = ind_array.reshape((ind_array.shape[0],-1), order='C')
    mpa_tabformat = np.array(mpa_list_single)[mov_inds]
    dc_tabformat = np.array(dc_list_single)[mov_inds]
    
    roi_labels = cell_idx[cell_inds]
    
    tab = pd.DataFrame({
                        'mov': mov_inds,
                        'roi': roi_labels,
                        'trial': trial_inds,
                        'dff_resp': dff_change.flatten(),
                        'spk_resp': spk_change.flatten(),
                        'mpa': mpa_tabformat,
                        'dc':dc_tabformat,
                        'resp_type': np.isin(cell_inds, responders).astype('int') - np.isin(cell_inds, neg_responders).astype('int')
                        })
    

    tab['mouse_reg'] = pd.Categorical( ['_'.join([mouse,region])]*len(cell_inds))
    
    # trial trace table
    
    dff_change = F.reshape((F.shape[0],len(mpa_list_single),-1,trial_length),order='C')[:,:,1:,:]
    dff_change = (dff_change - dff_change[:,:,:,prestim].mean(axis=-1, keepdims=True)) / dff_change.mean(axis=-1, keepdims=True)
    
    spk_change = spks.reshape((spks.shape[0],len(mpa_list_single),-1,trial_length),order='C')[:,:,1:,:]
    spk_change = (spk_change - spk_change[:,:,:,prestim].mean(axis=-1, keepdims=True)) /  spk_change.mean(axis=(2,3), keepdims=True)

    
    ind_array = np.indices(dff_change.shape) # cell_inds, mov_inds, trial_inds, frame_inds
    cell_inds, mov_inds, trial_inds, frame_inds = ind_array.reshape( (ind_array.shape[0],-1), order='C' )
    mpa_tabformat = np.array(mpa_list_single)[mov_inds]
    dc_tabformat = np.array(dc_list_single)[mov_inds]
    
    
    roi_labels = cell_idx[cell_inds]

    trial_tab = pd.DataFrame({
                        'mov': mov_inds,
                        'roi': roi_labels,
                        'trial': trial_inds,
                        'frame': frame_inds,
                        'dff_resp': dff_change.flatten(),
                        'spk_resp': spk_change.flatten(),
                        'mpa': mpa_tabformat,
                        'dc': dc_tabformat,
                        'resp_type': np.isin(cell_inds, responders).astype('int') - np.isin(cell_inds, neg_responders).astype('int')
                        })
    
    trial_tab['mouse_reg'] = pd.Categorical( ['_'.join([mouse,region])]*len(cell_inds))
    
    return tab, trial_tab


#######   #######
### fun APPLY ###
#######   #######


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

column_rename =  {
        'mpa':'Intensity (MPa)',
        'dff_resp': '\u0394F/F',
        'mouse_reg': 'mouse/region',
        'dc':'Duty cycle',
        'resp_type': 'Responder class'
}

responder_rename ={
    -1: 'negative',
    0: 'undefined',
    1: 'positive'
}

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