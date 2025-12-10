import os
import numpy as np
import glob
from tqdm import tqdm
import random
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error

from scipy.signal._peak_finding_utils import _select_by_peak_distance


def find_outliers(y, threshold=2):

    '''
    This function find outliers in e 1-d array based on the number of standard devitions form the mean
    y :  1-d array
    threshold : numbers of standard desviation to define outliers

    returns
    idx_outlier : boolean indexes indicating the outliers
    '''
    # remove nans
    y_ = y[np.isnan(y)==False]
    # define outliers 
    sd = np.std(y_)
    thresh_l = np.mean(y_) - threshold*np.std(y_)
    thresh_u = np.mean(y_) + threshold*np.std(y_)
    idx_outlier = np.logical_or(y<thresh_l, y>thresh_u)
    return idx_outlier


def remove_outliers(y, threshold=2):
    '''
    This function find outliers in e 1-d array based on the number of standard devitions form the mean
    y :  1-d array
    threshold : numbers of standard desviation to define outliers

    returns
    an array without the outliers
    '''
    idx_outlier = find_outliers(y, threshold=threshold)
    return y[idx_outlier==False]


def find_edges(x, max_transition_frames, limit, revert=False):
    '''
    This functions sees if in the array x the n=max_transition_frames frames at the edges are trasnition frames
    It checks whether by not considering them the array stays below the limits

    x : 1-d array 
    max_transition_frames : maximun number of trasnition frames (2*max_transition_frames must be < len(x))
    limit : limit to consider 
    revert : False uses the array as it is, True reverts the array

    returns 
    n_first : the number of edges to discart from the beging (or end if revert = True) to have x<limit
    '''
    
    if revert:
        x = np.flip(x)

    n_first = 0
    if len(x)>2*max_transition_frames:

        max_change = np.max(x[max_transition_frames:-max_transition_frames])
        if max_change<limit:
            n_first = max_transition_frames
            while n_first>=0:
                max_change = np.max(x[n_first:-max_transition_frames])
                if max_change<limit:
                    n_first-=1
                else:
                    break
            n_first = n_first+1

    return n_first


def find_peaks(y, window, distance=None, threshold=3, relative_threshold=True, min_thresh=4):
    '''
    This functions finds peaks in a one dimensional array when values are far from the previous and folowwing samples

    y  : one dimensional array 
    window : interger denoting the samples to take on both sides around a sample to detect whethere it is a peak
    distance : minimun peak distance
    threshold : either the threshold value (if relative_threshold is False) or the number of standard desviation used to determine the threhold (if relative_threshold is True)
    relative_threshold : boolean indicating whether to use or not relative thresholds
    min_thresh : only valid if the threshold is determined relatively. If the estimated threshold is smaller than this min_thresh is used

    returns
    peaks : indexes indicatig the positions of the peaks in y
    '''

    if distance is not None and distance < 1:
        raise ValueError('`distance` must be greater or equal to 1')
    
    peaks= []
    for i in range(len(y)):

        if not np.isnan(y[i]):

            # get the indexes to take the data before and after i
            idx_pre = np.arange(i-window, i)
            idx_pre = idx_pre[np.logical_and(idx_pre>=0, idx_pre<len(y))]
            idx_post = np.arange(i+1, i+1+window,1)
            idx_post = idx_post[np.logical_and(idx_post>=0, idx_post<len(y))]

            if len(idx_pre)>0 and len(idx_post)>0:
            
                # compute thresholds pre and post
                if relative_threshold:
                    y_no_outliers = remove_outliers(y[idx_pre], threshold=2)
                    if len(y_no_outliers)>2:
                        threshold_pre = np.mean(y_no_outliers) + threshold*np.std(y_no_outliers)
                        threshold_pre = max(threshold_pre, min_thresh)

                    else:
                        threshold_pre = None
                    
                    y_no_outliers = remove_outliers(y[idx_post], threshold=2)
                    if len(y_no_outliers)>2:
                        threshold_post = np.mean(y_no_outliers) + threshold*np.std(y_no_outliers)
                        threshold_post = max(threshold_post, min_thresh)
                    else:
                        threshold_post = None
                else:
                    threshold_pre = threshold
                    threshold_post = threshold

                # determine if it is peak  
                if threshold_pre is not None:
                    is_pre = y[i] > threshold_pre
                else:
                    is_pre = False
                if threshold_post is not None:
                    is_post = y[i] > threshold_post
                else:
                    is_post = False
                if is_pre and is_post:
                    peaks.append(i)

    # conver to array
    peaks = np.array(peaks)    
    
    # remove close peaks
    if distance is not None and len(peaks)>1:
        keep = _select_by_peak_distance(peaks, np.float64(y[peaks]), np.float64(distance))
        peaks = peaks[keep]

    return peaks


def find_margin(data, limit=0, axis=0, revert=False):

    """
    This function finds the number of pixels that can be taken from the dimension selected an have range of intensity lower than a limit
    data : 3-d array (last dimension are the frame number)
    limit : upper limit to the intensity range within d[:m,:,:]
    axis : the axis to consider
    revert : whether to flip or not the data first

    returns the number of pixels such that d[:m,:,:] has a range lower than limit
    """
    if axis==1:
        data = np.transpose(data,(1,0,2))
    if revert:
        data = np.flip(data, axis=0)
    m = 0
    while m < np.shape(data)[0]:
        if (np.max(data[:m+1,:,:])-np.min(data[:m+1,:,:]))<=limit:
            m+=1
        else:
            break
    return m


class Videos:

    """
    This class handles the videos. Loads, plot, find segments, classifies 
    """
    
    # Parameter for the limit for the intensity range to define something is spacially uniform 
    limit_intensity_range = 15
    # Parameter limitating the maxium accepted change to define that segmetns are static
    limit_no_change = 20

    def __init__(self, folder, video):
        self.recording = os.path.basename(folder)
        self.video = video
        self.file_name = os.path.basename(video)
        self.data = np.load(os.path.join(folder, 'data', 'videos', video))

    def get_valid_frames(self):
        """
        It gets the number of valid frames (not nans) in the video
        """
        n_emptyframes = np.sum(np.all(np.all(np.isnan(self.data),axis=0),axis=0))
        self.valid_frames = np.shape(self.data)[2]-n_emptyframes
        return self.valid_frames
    
        
    def compute_time_change(self):
        """
        It computes the differences between consecutive frames using mean squared error and stores it in self.changes
        """
        if not hasattr(self, 'valid_frames'):
            self.get_valid_frames()
        change = np.zeros(self.valid_frames-1)
        for i in range(self.valid_frames-1):
            change[i] = mean_squared_error(self.data[:,:,i+1], self.data[:,:,i])
        return change
    
    def find_peaks(self, limit=limit_no_change):

        """
        It finds the peaks in the changes and stores it in self.peaks
        """

        change = self.compute_time_change()

        window = 10
        threshold_peaks = 5
        peaks = find_peaks(change, window, distance=7, threshold=threshold_peaks, relative_threshold=True, min_thresh=10)

        # store in the object
        self.peaks = peaks 
        self.n_peaks = len(peaks)
    

    def define_segments(self):

        '''
        It finds the video segments (start, end, duration) based on the peaks and stores it in self.segments
        self.segments is a dictionary
        '''

        # find the peaks
        if not hasattr(self, 'peaks'):
            self.find_peaks()

        if not hasattr(self, 'valid_frames'):
            self.get_valid_frames()


        # initialize the segments dictionary
        self.n_segments = self.n_peaks+1
        segments = {'frame_start':np.empty(self.n_segments, dtype=int), 
                    'frame_end': np.empty(self.n_segments, dtype=int), 
                    'duration':np.empty(self.n_segments, dtype=int),
                    }
        
        # peaks initial and last sample
        if self.n_peaks>0:
            previous_peak = 0
            for k, ipeak in enumerate(self.peaks):
                segments['frame_start'][k] = previous_peak
                segments['frame_end'][k] = ipeak+1
                previous_peak = ipeak+1
            segments['frame_start'][k+1] = previous_peak
            segments['frame_end'][k+1] = self.valid_frames
        else:
            segments['frame_start'][0] = 0
            segments['frame_end'][0] = self.valid_frames

        # compute the segments duration
        for idx, (ki,kf) in enumerate(zip(segments['frame_start'], segments['frame_end'])):
            segments['duration'][idx] = kf-ki
             
        # store it into the object
        self.segments = segments

        return segments


    def find_static_segments(self, limit=limit_no_change, max_transition_frames=3):

        '''
        It finds the maximun change in each segments and defines segments as static if the maximun change is < limit
        Since when static frames are present there might be transition frames. It tries to identity them but this doesn not works very weel yet.
        Then, the computation of the maximun is done skiping the 2 first and 2 last frames in the segment
        '''

        if not hasattr(self, 'segments'):
            self.define_segments()
        
        change = self.compute_time_change()
        
        # initialize
        self.segments['max_change'] = np.empty(self.n_segments, dtype=float)
        self.segments['is_static'] = np.empty(self.n_segments, dtype=bool)
        self.segments['transition_start'] = np.empty(self.n_segments, dtype=int)
        self.segments['transition_end'] = np.empty(self.n_segments, dtype=int)

        # define which segment are a static picture. There might be transition frames in the case of static images
        for idx, (ki,kf) in enumerate(zip(self.segments['frame_start'],self.segments['frame_end'])):
            
            # idx_non_zero = change[ki:kf]>limit
            # n_first = find_first_n_true(idx_non_zero, revert=False)
            # n_last = find_first_n_true(idx_non_zero, revert=True)
            # # if too many trasnition segments, then do no consider it
            # max_trans = min(max_transition_frames, int(0.1*self.segments['duration'][idx]))
            # if n_first>max_trans or n_last>max_trans:
            #     n_first = 0
            #     n_last = 0
            
            # # find the maximun change without counting 2 frames in the edges that might be transition frames 
            # if self.segments['duration'][idx] > (2*max_transition_frames):
            #     self.segments['max_change'][idx] = np.max(change[ki+2:kf-2])
            # else:
            #     self.segments['max_change'][idx] = np.max(change[ki:kf])

            # # store the descriptions
            # is_static = False
            # if n_first<(kf-ki) and n_last<(kf-ki):
            #     if self.segments['max_change'][idx]<=limit:
            #         self.segments['is_static'][idx] = True
            #         self.segments['transition_start'][idx] = n_first
            #         self.segments['transition_end'][idx] = n_last
            #         is_static = True
            # if not is_static:
            #     self.segments['is_static'][idx] = False
            #     self.segments['transition_start'][idx] = 0
            #     self.segments['transition_end'][idx] = 0


            # find the maximun change onsidering there might be a maximun of 3 frames in the edges that might be transition frames (common in the case of static pictures)
            if self.segments['duration'][idx] > (2*max_transition_frames):
                n_first = find_edges(change[ki:kf], max_transition_frames, limit, revert=False)
                n_last = find_edges(change[ki:kf], max_transition_frames, limit, revert=True)
                self.segments['max_change'][idx] = np.max(change[ki+n_first:kf-n_last])
            else:
                self.segments['max_change'][idx] = np.max(change[ki:kf])
                n_first = 0
                n_last = 0

            # store the descriptions
            self.segments['is_static'][idx] = self.segments['max_change'][idx]<=limit
            self.segments['transition_start'][idx] = n_first
            self.segments['transition_end'][idx] = n_last
    

    def find_segments_intensity_range(self):

        """
        It finds the range for the intensity in that video
        """

        if not hasattr(self, 'segments'):
            self.define_segments()
        
        # initialize
        self.segments['intensity_range'] = np.empty(self.n_segments, dtype=float)

        # compute
        for idx in range(len(self.segments['frame_start'])):
            ki = self.segments['frame_start'][idx]+self.segments['transition_start'][idx] 
            kf = self.segments['frame_end'][idx]-self.segments['transition_end'][idx] 
            self.segments['intensity_range'][idx] = np.max(self.data[:,:,ki:kf]) - np.min(self.data[:,:,ki:kf])

    

    def find_margins_segments(self, limit=limit_intensity_range):

        """
        It finds the margin for the segments
        """
        if not hasattr(self, 'segments'):
            self.define_segments()

        if 'intensity_range' not in self.segments.keys():
            self.find_segments_intensity_range()
        
        self.segments['margin_left'] = np.empty(self.n_segments, dtype=int)
        self.segments['margin_rigth'] = np.empty(self.n_segments, dtype=int)
        self.segments['margin_top'] = np.empty(self.n_segments, dtype=int)
        self.segments['margin_bottom'] = np.empty(self.n_segments, dtype=int)
        for idx, (ki,kf) in enumerate(zip(self.segments['frame_start'],self.segments['frame_end'])):

            if self.segments['intensity_range'][idx]<=limit:
                self.segments['margin_left'][idx] = np.shape(self.data)[1]
                self.segments['margin_rigth'][idx] = np.shape(self.data)[1]
                self.segments['margin_top'][idx] = np.shape(self.data)[0]
                self.segments['margin_bottom'][idx] = np.shape(self.data)[0]
                
            else:
                self.segments['margin_top'][idx] = find_margin(self.data[:,:,ki:kf], limit=limit, axis=0, revert=False)
                self.segments['margin_bottom'][idx] = find_margin(self.data[:,:,ki:kf], limit=limit, axis=0, revert=True)
                self.segments['margin_left'][idx] = find_margin(self.data[:,:,ki:kf], limit=limit, axis=1, revert=False)
                self.segments['margin_rigth'][idx] = find_margin(self.data[:,:,ki:kf], limit=limit, axis=1, revert=True)
                

    def find_segment_brackground(self):
        """
        It finds percentual of pixels in the video occupied by backgound
        """
        self.segments['backgound_proportion'] = np.empty(self.n_segments, dtype=float)
        for idx, (ki,kf) in enumerate(zip(self.segments['frame_start'],self.segments['frame_end'])):
            d = self.data[:,:,ki:kf]
            hist, bin_edges = np.histogram(d, bins=255, range=(0,256), density=True)
            self.segments['backgound_proportion'][idx] = np.max(hist)


    def describe_segments(self):
        """
        It runs all the methods describing the segments
        """
        self.find_static_segments()
        self.find_segments_intensity_range()
        self.find_margins_segments()
        self.find_segment_brackground()
        

    def print_segments_table(self):
        """
        It prints a table describing the segments
        """
        df = pd.DataFrame(self.segments)
        return df

    def is_gaussiandot(self):
        """
        This methods detemrine whetehr the video satisfies the coditions to be a GaussianDot video
        """
        
        # parameters
        duration = 9  # expected duration in frames
        tolerance = 1 # tolerance to consider good duration 
        background_thresh = 0.5 # minimun proprotion of background
        p_limit = 0.8 # minimun required proportion of segments matching the requierments    

        # check if the segments are static
        p_static = np.sum(self.segments['is_static'])/len(self.segments['is_static'])

        # check if the segments have the good duration
        good_duration = np.logical_and( self.segments['duration']>=(duration-tolerance),  self.segments['duration']<=(duration+tolerance))
        p_good_duration = np.sum(good_duration)/len(good_duration)

        # check if all have background
        good_background = self.segments['backgound_proportion']>=background_thresh
        p_good_background = np.sum(good_background)/len(good_background)

        # decide whether it is a GaussianDot
        if p_static>=p_limit and p_good_duration>=p_limit and p_good_background>=p_limit:
            is_gaussian = True
        else:
            is_gaussian = False

        return is_gaussian
    
    
    def is_naturalimage(self, limit_intensity_range=limit_intensity_range):
        """
        This methods detemrine whetehr the video satisfies the coditions to be a NaturalImage video
        """

        # parameters
        duration_black = [12, 18]  # expected duration black images
        duration_image = 15  # expected duration image
        tolerance = 2     # tolerance to consider good duration
        background_thresh = 0.5 # minimun proprotion of background
        p_limit = 0.8 # minimun required proportion of segments matching the requierments    
             
        # check sequence structure
        if any(self.segments['intensity_range']<=limit_intensity_range):

            # check if the segments are static
            p_static = np.sum(self.segments['is_static'])/len(self.segments['is_static'])

            # # check alternation of black and immages
            # first_black = np.where(self.segments['intensity_range']<=limit_intensity_range)[0][0]
            # if first_black==0:
            #     first_immage = 1
            # else:
            #     first_immage = 0
            
            # idx = np.arange(first_black,self.n_segments,2)
            # alternate_black = np.all(self.segments['intensity_range'][idx]<=limit_intensity_range)
            # idx = np.arange(first_immage,self.n_segments,2)
            # alternate_immage = np.all(self.segments['intensity_range'][idx]>limit_intensity_range)
            # alternate = alternate_black and alternate_immage

            # check alternation of black and immages
            first_black = np.where(self.segments['backgound_proportion']>=background_thresh)[0][0]            
            idx = np.arange(first_black,self.n_segments,2)
            alternate_black = self.segments['backgound_proportion'][idx]>=background_thresh
            p_alternate_black = np.sum(alternate_black)/len(alternate_black)

            # check duration
            dur_black = self.segments['duration'][self.segments['intensity_range']<=limit_intensity_range]
            dur_image = self.segments['duration'][self.segments['intensity_range']>limit_intensity_range]
            good_duration_balck = np.logical_and( dur_black>=(duration_black[0]-tolerance),  dur_black<=(duration_black[1]+tolerance))
            good_duration_immage = np.logical_and( dur_image>=(duration_image-tolerance),  dur_image<=(duration_image+tolerance))
            p_good_duration = (np.sum(good_duration_balck) + np.sum(good_duration_immage)) /(len(good_duration_balck)+len(good_duration_immage))
                    
            # decide whether it is a Natural Image
            if p_static>=p_limit and p_good_duration>=p_limit and p_alternate_black>=p_limit:
                is_naturalimage = True
            else:
                is_naturalimage = False
        
        else:
            is_naturalimage = False

        return is_naturalimage
    

    def is_gabor(self):
        """
        This methods detemrine whetehr the video satisfies the coditions to be a Gabor video
        """
        
        # parameters
        duration = 25
        tolerance = 2  
        min_margin_size = 5
        background_thresh = 0.5 # minimun proprotion of background
        p_limit = 0.8 # minimun required proportion of segments matching the requierments    

        # check if the segments have the good duration
        good_duration = np.logical_and( self.segments['duration']>=(duration-tolerance),  self.segments['duration']<=(duration+tolerance))
        p_good_duration = np.sum(good_duration)/len(good_duration)

        # check it has margins
        marg = ['margin_left','margin_rigth']
        good_marg = True
        for margi in marg:
            good_marg = good_marg and np.all(self.segments[margi]>=min_margin_size)

        # check if all have background
        good_background = self.segments['backgound_proportion']>=background_thresh
        p_good_background = np.sum(good_background)/len(good_background)

        
        # decide whether it ia a Gabor
        if p_good_duration>=p_limit and good_marg>=p_limit and p_good_background>=p_limit:
            is_gabor = True
        else:
            is_gabor = False

        return is_gabor
    

    def is_pinknoise(self):
        """
        This methods detemrine whetehr the video satisfies the coditions to be a PinkNoise video
        """
        
        # parameters
        duration = 27
        tolerance = 2  
        min_margin_size = 5
        p_limit = 0.8 # minimun required proportion of segments matching the requierments    

        # check if the segments have the good duration
        good_duration = np.logical_and( self.segments['duration']>=(duration-tolerance),  self.segments['duration']<=(duration+tolerance))
        p_good_duration = np.sum(good_duration)/len(good_duration)
        
        # check it does not have margin (it can be mixed with Gabor that does have it)
        marg = ['margin_left','margin_rigth']
        marg_left = self.segments['margin_left']>=min_margin_size
        marg_right = self.segments['margin_rigth']>=min_margin_size
        has_marg = marg_right = self.segments['margin_rigth']>=min_margin_size
        no_has_marg = has_marg==False
        p_no_has_marg = np.sum(no_has_marg)/len(no_has_marg)

        # has_marg = False
        # for margi in marg:
        #     has_marg = has_marg or np.all(self.segments[margi]>=min_margin_size)
        # no_has_marg = has_marg==False
        # p_no_has_marg = np.sum(no_has_marg)/len(no_has_marg)

        # decide whether it is Pink noise
        if p_good_duration>=p_limit and p_no_has_marg>=p_limit:
            is_pinknoise = True
        else:
            is_pinknoise = False

        return is_pinknoise
    

    def is_randomdots(self):
        """
        This methods detemrine whetehr the video satisfies the coditions to be a RandomDots video
        """
        
        # parameters
        duration = 60
        tolerance = 2 
        min_proportion_background = 0.5
        p_limit = 0.8 # minimun required proportion of segments matching the requierments    


        # check if the segments have the good duration
        good_duration = np.logical_and( self.segments['duration']>=(duration-tolerance),  self.segments['duration']<=(duration+tolerance))
        p_good_duration = np.sum(good_duration)/len(good_duration)

        # check there is a backgound
        good_background  = self.segments['backgound_proportion']>=min_proportion_background
        p_good_background = np.sum(good_background)/len(good_background)
        
        # decide whether it ia a Gabor
        if p_good_duration>=p_limit and p_good_background==1:
            is_randomdot = True
        else:
            is_randomdot = False

        return is_randomdot
        

    def is_naturalvideo(self, lim_segments=3, min_var_duration=1, limit_intensity_range=limit_intensity_range):
        """
        This methods detemrine whetehr the video satisfies the coditions to be a NaturalVideo video
        """       
        
        non_uniform = np.all(self.segments['intensity_range']>limit_intensity_range)
        
        if self.n_segments<=lim_segments:
            no_pattern = True
        else:
            if np.std(self.segments['duration'])>=min_var_duration:
                no_pattern = True
            else:
                no_pattern = False

        if non_uniform and no_pattern:
            is_naturalvideo = True
        else:
            is_naturalvideo = False

        return is_naturalvideo
    
    def classify(self):
        """
        This methods classities the video. It tries to see wheter it is any of the classes besides NaturalVideo. 
        If only one satisfies the conditions, then, that is selected
        If more then one satisfies the conditions, then the label is set as "unknonw"
        If none satisfies the conditions, an it is NaturalVideo
        """

        # labels to check first, if noone of this check if it can be a Natural Video
        labels_ = {'GaussianDot':'is_gaussiandot',
                    'NaturalImages':'is_naturalimage',
                    'Gabor':'is_gabor',
                    'PinkNoise':'is_pinknoise',
                    'RandomDots':'is_randomdots',
                    }
        # check whether it could be the labels in labels_
        res = {}
        for labi,methodi in labels_.items():
            c = getattr(self, methodi)
            res[labi] = c()
        # check if only one was label matched in labels_, if more than one set as unknownw, if none check if it can be a NaturalVideo
        k = np.array(list(res.keys()))
        v = np.array(list(res.values()))
        if len(np.where(v)[0])==1:
            label = k[np.where(v)[0][0]]
        else:
            if self.is_naturalvideo():
                label = 'NaturalVideo'
            else:
                label = 'unknown'

        self.label = str(label)
        return str(label)
        

    def plot_changes(self):
        """
        It plots the changes and the peaks for the video
        """
        # compute change
        change = self.compute_time_change()
        # find the peaks
        if not hasattr(self, 'peaks'):
            self.find_peaks()
        # plot
        fig, ax = plt.subplots(1,1)
        ax.plot(change[:self.valid_frames-1], 'k.', linestyle='-')
        if self.n_peaks>0:
            ax.plot(self.peaks, change[self.peaks],'rx', linestyle='none')
        return fig, ax
        

    def plot_intensity_hist_all(self):
        """
        It plots an histogram with the distribution of the intensities for all the video
        """
        img = self.data
        fig, ax = plt.subplots(1,1)
        ax.hist(img.flatten(), range=(0, 255), bins=255)
        return fig, ax


    def plot_intensity_hist(self, segment):
        """
        It plots an histogram with the distribution of the intensities for the specified segment
        """
        ki = self.segments['frame_start'][segment]+self.segments['transition_start'][segment]
        kf = self.segments['frame_end'][segment]-self.segments['transition_end'][segment]
        img = self.data[:,:,ki:kf]
        fig, ax = plt.subplots(1,1)
        ax.hist(img.flatten(), range=(0, 255), bins=255)
        return fig, ax


    def plot_frame(self, frame):
        """
        It plots the frame indicated by frame
        """
        fig, ax =plt.subplots(1,1)
        ax.imshow(self.data[:,:,frame], cmap='gray')
        return fig, ax


    def plot_frames(self, frames):
        """
        It plots the frames indicated by frame
        """
        n_frames = len(frames)
        ncol = 5
        nrow = int(n_frames//ncol)
        if ncol<n_frames/nrow:
            nrow = (n_frames//ncol)+1
        fig, ax = plt.subplots(nrows=nrow, ncols=ncol)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        for j, i in enumerate(frames):
            ax.flatten()[j].imshow(self.data[:,:,i], cmap='gray', vmin=0, vmax=255)
            ax.flatten()[j].set_title(f"frame {i}")
        return fig, ax
                    

    
    