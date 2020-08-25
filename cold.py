import pandas as pd
import numpy as np
from makeModel import MakeCCDCModel
import fcntl
import csv

def setupModels(all_band_data, bands, init_obs, alpha):
    
    """Creates a model for each band and stores it in model_list"""

    # Get column of datetime values
    datetimes = all_band_data[:,0]
    
    # Create a model for each band and store it in model_list
    for i in range(1, all_band_data.shape[1]):
        
        band_data = all_band_data[:,i]
   
        ccdc_model = MakeCCDCModel(datetimes, init_obs, bands[i-1])
            
        ccdc_model.fitModel(band_data, alpha)
        
        model_list[i-1] = ccdc_model
        
    #print("Models initialized")

def getNumYears(date_list):

    """Get number of years spanned by the dataset (from Python/Rata Die date)"""
    
    try:
        last_date = np.amax(date_list)
        first_date = np.amin(date_list)
            
        num_years = (last_date - first_date) / 365.25
        
    except ValueError as err:
        print("ValueError when trying to find number of years covered by dataset: {}".format(err))
        print(date_list)
        
        return 0

    return num_years

def datesToNumbers(dates):
    
    dates_as_ordinal = np.array([pd.Timestamp(x).toordinal() for x in dates])
    
    return dates_as_ordinal

def transformToArray(dataset_to_transform):

    """Transforms xarray Dataset object into a Numpy array"""
    
    ds_to_array = datesToNumbers(dataset_to_transform.time.data).reshape(-1, 1)
    
    for var in dataset_to_transform.data_vars:
        ds_to_array = np.hstack((ds_to_array, dataset_to_transform[var].values.reshape(-1, 1)))
        
    # Remove NaNs and sort by datetime    
    ds_to_array = tidyData(ds_to_array)
        
    return ds_to_array 

def tidyData(pixel_ts):
    
    """Takes a single pixel time series, removes NaN values, and sorts by date."""
     
    # Remove NaNs
    pixel_nan_mask = np.any(np.isnan(pixel_ts), axis=1)
    pixel_ts = pixel_ts[~pixel_nan_mask]
    
    # Sort by date
    pixel_ts = pixel_ts[np.argsort(pixel_ts[:,0])]
                                             
    return pixel_ts

def writeRows(fp, rows):
    
    with open(fp, 'a') as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        writer = csv.writer(output)
        writer.writerows(rows)   
        fcntl.flock(output, fcntl.LOCK_UN)
   
def initModel(pixel_data, bands, init_obs, alpha):

    """Finds a sequence of 6/12/18/24 consecutive clear observations without any change, to initialize the model"""

    #print("Starting initialization...")
    
    num_bands = len(bands)
    
    # Subset first n clear observations for model initialisation 
    curr_obs_list = pixel_data[:init_obs,]

    # Start off with the model uninitialized
    model_init = False
    num_iters = 0
    
    # The next observation to be added to the model to detect change
    init_end = None

    # Model initialization sequence - keeps going until a clear set of observations is found
    while(model_init == False):
        
        #print("Trying to initialize...")

        # Check if there are not enough data points left to initialise the models
        num_data_points = len(curr_obs_list)
        
        if(num_data_points < init_obs):
            #print("Could not find a period of no change for model initialization.")
            return None
        
        init_years = getNumYears(curr_obs_list[:,0]) # Select first column (dates)
        
        extra_vals = 0
        
        while(init_years < 1):
            
            #print("Adding points to get 1 year of data...")
            
            extra_vals += 1
            
            curr_obs_list = pixel_data[num_iters:init_obs+num_iters+extra_vals,] # Add an observation
            
            init_years = getNumYears(curr_obs_list[:,0])
            
            # Check if we have reached the end of the remaining data and covered < 1 year
            if((curr_obs_list[-1, 0] == pixel_data[-1, 0]) and init_years < 1):
                return None
 
        # Re-initialize the models
        setupModels(curr_obs_list, bands, init_obs, alpha)

        # Get total time used for model initialization
        total_time = np.max(curr_obs_list[:,0]) - np.min(curr_obs_list[:,0])
        
        total_slope_eval = 0
        total_start_eval = 0
        total_end_eval = 0
        
        # Check for change during the initialization period. We need 12 observations with no change
        for band_model in model_list: # For each model

            slope_val = np.absolute(band_model.coefficients[0]) / (3 * band_model.RMSE / total_time)
            total_slope_eval += slope_val
        
            start_val = np.absolute(band_model.start_val - band_model.predicted[0]) / (3 * band_model.RMSE)
            total_start_eval += start_val
            
            end_val = np.absolute(band_model.end_val - band_model.predicted[init_obs-1]) / (3 * band_model.RMSE)
            total_end_eval += end_val
            
        total_slope_eval = np.abs(total_slope_eval / num_bands)
        total_start_eval = np.abs(total_start_eval / num_bands)
        total_end_eval = np.abs(total_end_eval / num_bands)
 
        # Test if not stable
        if(total_slope_eval + np.max([total_start_eval, total_end_eval]) > 0.99):
            num_iters += 1
            curr_obs_list = pixel_data[num_iters:init_obs+num_iters,:] # Not stable; Shift along 1 row
        
        else:
            model_init = True
            init_end = init_obs + num_iters + extra_vals
            #print("Model initialized. Iterations needed: {}".format(num_iters))

    return curr_obs_list, init_end

def jumpAhead(pixel_data, bands, next_obs, alpha):

    start_ix = next_obs
    
    num_bands = len(bands)
    
    while(True):
        
        end_ix = start_ix + 24

        curr_window = pixel_data[start_ix:end_ix] # Window we are fitting the new model to
        
        if(len(curr_window) < 24):
            return pixel_data[:start_ix,], start_ix
        
        # Get column of datetime values
        datetimes = curr_window[:,0]
            
        new_models = []
        
        # Create a model for each band and store it in model_list
        for i in range(1, num_bands+1):
    
            band_data = curr_window[:,i]
    
            ccdc_model = MakeCCDCModel(datetimes, 24, bands[i-1], alpha)
    
            ccdc_model.fitModel(band_data)
            
            new_models.append(ccdc_model)
    
        # Get total time used for model initialization
        total_time = np.max(curr_window[:,0]) - np.min(curr_window[:,0])
        
        total_slope_eval = 0
        total_start_eval = 0
        total_end_eval = 0
        
        # Check for change during the initialization period
        for band_model in new_models: # For each new model
            
            slope_val = np.absolute(band_model.coefficients[0]) / (3 * band_model.RMSE / total_time)
            total_slope_eval += slope_val
        
            start_val = np.absolute(band_model.start_val - band_model.predicted[0]) / (3 * band_model.RMSE)
            total_start_eval += start_val
            
            end_val = np.absolute(band_model.end_val - band_model.predicted[23]) / (3 * band_model.RMSE)
            total_end_eval += end_val
     
        # If model is stable
        if((total_slope_eval / num_bands) < 1 and (total_start_eval / num_bands) < 1 and (total_end_eval / num_bands) < 1):
    
            change_eval = 0
            
            # Check for similarity to the first model over all bands
            for i in range(1, num_bands+1): # For each band
                
                rmse = model_list[i-1].RMSE # RMSE of model fitted to previous 24 observations
                
                real_vals = curr_window[:,i] # Next 24 observations
    
                predicted_vals = model_list[i-1].getPrediction(datetimes) # Predicted values for this time period
            
                # Calculate RMSE for the new observations
                error = real_vals - predicted_vals
                
                new_rmse = np.sqrt(np.mean(error**2))
                
                change_eval += (new_rmse / (3*rmse))

            if((change_eval / num_bands) <= 1): # All bands show enough similarity to the previous models
                
                # Add the data from the window to the model
                setupModels(pixel_data[:end_ix,], bands, 24, alpha)
                start_ix += 24
                
            else:
                return pixel_data[:start_ix,], start_ix
                
        else: # Not stable period, so can't be added to the model
            return pixel_data[:start_ix,], start_ix
        
def findChange(pixel_data, bands, init_obs, output_file, use_temporal, re_init, ch_thresh, alpha, rows, x=None, y=None):
    
    """Continues to add data points to the model until either a new breakpoint is detected, or there
        are not enough observations remaining."""
        
    num_bands = len(bands)

    # Initialise model
    try:
        model_data, next_obs = initModel(pixel_data, bands, init_obs, alpha)
    except TypeError:
        return []
    
    if(use_temporal):
        # See if model can be extended to reduce run time
        if(len(pixel_data) >= 48 and init_obs == 24):
            try:
                #print('Trying jump...')
                model_data, next_obs = jumpAhead(pixel_data, bands, next_obs, alpha)
            except TypeError:
                #print('Jump failed')
                pass
    
    if(x):
        model_output = [[x, y, m.band, m.getMinDate(), m.getMaxDate(), m.start_val, m.end_val, ["{:0.5f}".format(c) for c in m.coefficients], m.RMSE, m.lasso_model.intercept_] for m in model_list]
    else:
        model_output = [[m.band, m.getMinDate(), m.getMaxDate(), m.start_val, m.end_val, ["{:0.5f}".format(c) for c in m.coefficients], m.RMSE, m.lasso_model.intercept_] for m in model_list]

    # Detect change
    change_flag = 0
    change_date = None
    change_mags = None
    
    prev_date = None
    
    change_vectors = []
    
    #print("Running...")

    while((next_obs+1) <= len(pixel_data)):

        # Get next row from the time series
        new_obs = pixel_data[next_obs,]
        
        # Get date
        new_date = new_obs[0]
        
        if not prev_date:
            prev_date = new_date
          
        # Predicted value for the new observation in each band
        pred_obs = [model_list[i].getPrediction(new_date)[0] for i in range(num_bands)]
        
        # Actual - predicted for each band
        diff_obs = [new_obs[i] - pred_obs[i-1] for i in range(1, num_bands+1)]
        
        # Normalize by RMSE
        v_diff = [(diff_obs[i] / model_list[i].getRMSE(new_date)) for i in range(num_bands)]
        
        # Get vector magnitude
        vec_mag = np.linalg.norm(v_diff)**2
            
        if(vec_mag < ch_thresh): # No deviation from model detected
            
            #print("Adding new data point")
            model_data = np.append(model_data, [new_obs], axis=0)
            
            date_diff = new_date - prev_date # Will be 0 on first pass
            
            if(date_diff >= re_init):
                setupModels(model_data, bands, init_obs, alpha)
                
                if(x):
                    model_output = [[x, y, m.band, m.getMinDate(), m.getMaxDate(), m.start_val, m.end_val, ["{:0.5f}".format(c) for c in m.coefficients], m.RMSE, m.lasso_model.intercept_] for m in model_list]
                else:
                    model_output = [[m.band, m.getMinDate(), m.getMaxDate(), m.start_val, m.end_val, ["{:0.5f}".format(c) for c in m.coefficients], m.RMSE, m.lasso_model.intercept_] for m in model_list]

                prev_date = new_date
                
            change_flag = 0 # Reset change flag because we have an inlier
            change_vectors = []

        else: # Deviation from model detected
            
            #print("Outlier detected")
            
            change_flag += 1 # Don't add the new pixel to the model, but log possible change
            
            if(change_flag == 1): # If this is the first observed possible change point
                change_date = new_date # Log this as the new possible date of change
                change_mags = v_diff # Record change vectors
                
            change_vectors.append(v_diff)

        if(change_flag == 6):
            
            #print("Change detected!")
            
            # We have six consecutive deviating values
            
            # Calculate angles between consecutive changed pixels
            angles = [np.arccos((np.dot(change_vectors[i], change_vectors[i+1])) / (np.linalg.norm(change_vectors[i]) * np.linalg.norm(change_vectors[i+1]))) * 180 / np.pi for i in range(5)]

            # Get mean angle
            mean_ang = np.mean(angles)
            
            if(mean_ang <= 45): # Use threshold of 45 degrees
                
                # Change flagged
                # Check for false change due to regrowth
                
                for model_ix in range(len(model_output)):
                    model_output[model_ix].append(change_date) # Add date of change to output
                    model_output[model_ix].append(change_mags[model_ix])
                    rows.append(model_output[model_ix])
                                            
                return pixel_data[next_obs-5:,] # Return index of date when change was first flagged
            
            else:    
                # Mean angle not great enough to confirm change; discard observations
                change_flag = 0 
                change_vectors = []
                
        # Need to get the next observation
        next_obs += 1   
    
    # No change detected, end of data reached
    for model_ix in range(len(model_output)):
        rows.append(model_output[model_ix])
    
    #print("Done")
    return []
    
def runCOLD(input_data, bands, output_file, use_temporal, re_init, ch_thresh, alpha, x=None, y=None):

    """The main function which runs the CCDC algorithm. Loops until there are not enough observations
        left after a breakpoint to attempt to initialize a new model."""

    rows = [] 
    
    global model_list
    model_list = [None for i in range(len(bands))] # Set up list of current models
    
    # Get number of years covered by time series
    total_years = getNumYears(input_data[:,0])

    # The algorithm needs at least 12 observations to start, covering at least 1 year
    if(len(input_data) >= 12 and total_years >= 1):

        # Get total number of clear observations for this pixel
        num_clear_obs = len(input_data)
        
        # Decide window size - this is based on a minimum of 6 obs, plus 6 to detect change
        # A larger window size means a more complex model will be fitted
        
        if(num_clear_obs >=12 and num_clear_obs < 18):
            # Use simple model with initialization period of 6 obs
            window = 6
        
        elif(num_clear_obs >= 18 and num_clear_obs < 24):
            # Use simple model with initialization period of 12 obs
            window = 12

        elif(num_clear_obs >= 24 and num_clear_obs < 30):
            # Use advanced model with initialization period of 18 obs
           window = 18
        
        elif(num_clear_obs >= 30):
            # Use full model with initialisation period of 24 obs
            window = 24
        
        # Remaining data length needs to be smaller than window size
        while(len(input_data) >= window):
            
            if(x):
                input_data = findChange(input_data, bands, window, output_file, use_temporal, re_init, ch_thresh, alpha, rows=rows, x=x, y=y)
            else:
                input_data = findChange(input_data, bands, window, output_file, use_temporal, re_init, ch_thresh, alpha, rows=rows)

    if(x): # Write results out to file    
        writeRows(output_file, rows)
    if not (x):
        return rows
                    
            
                                                                                              