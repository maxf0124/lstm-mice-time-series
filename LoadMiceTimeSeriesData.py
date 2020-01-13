'''
LoadMiceTimeSeriesData is used to load and clean tamu time series data. The purpose is to use the processed data to
train BiDirectionalLSTM model for classification(as of now). (TODO: GIVE AN EXAMPLE)
'''
import calcom.io.CCDataSet as ccLoad
import numpy as np
import pdb

class MiceLoader:
    def __init__(self):
        # CCLines: TAMU mice lines
        self.CCLines = None
        # time_interval: Length of each segment
        self.time_interval = None
        # data dictionary to hold CCLines data
        self.data_dict = None
        # CC dataset
        self.ccd = None
        # full time series
        self.data_dict_ts = None
        # fft smooth coefficients
        self.num_coeffs = 100

    def load_ccd(self, path):
        self.ccd = ccLoad(path)

    def get_ccDataInDict(self,CCLines, time_interval, embedding):
        '''
        :param CCLines: A list of CCLine names
        :param time_interval: time interval for each sample
        :param embedding: [dim, delay] dim: integer embedding dimention delay: delay time
        :return: data_dict: A dictionary and its key values are mouse names
                 each item is a list: [Data,Label,Interval]
                 Data: (#samples,#time-steps) 2-d array
                 Label: (#samples,2) 2-d list: pre-inoculation[1,0] or after-inoculation[0,1]
                 Interval: (#samples, time_interval) 2-d list: [start:end] start to end of the time series relative to
                            inoculation time
        '''
        self.CCLines = CCLines
        self.time_interval = time_interval
        self.embed_dim, self.delay = embedding
        # get all idx corresponding to given CCLines
        lineIdx = self.ccd.find_attr_by_value('line',self.CCLines)
        # generate dictionary
        self.data_dict = {name: self.get_mice_data(idx)
                          for name,idx in zip(self.ccd.get_attr_values('mouse_id',idx=lineIdx),lineIdx)}
        return self.data_dict

    def get_ccDataInDict_full(self,CCLines):
        '''
        This function takes in CClines and return dictionary contains the full time series data
        :param CCLines: A list of CC lines
        :return: dict_ts: A dictionary: key values are mouse names, each item is a 1-d array(full time series corresponding to that mouse)
        '''
        self.CCLines = CCLines
        lineIdx = self.ccd.find_attr_by_value('line', self.CCLines)
        self.data_dict_ts = {name: self.get_full_time_series(idx) for name,idx in zip(self.ccd.get_attr_values('mouse_id',idx=lineIdx),lineIdx)}
        return self.data_dict_ts

    def get_mice_data(self,idx):
        '''
        This function takes in an index in self.ccd and output list: [Data,Label,Interval]
                 Data: (#samples,#time-steps) 2-d array
                 Label: (#samples,2) 2-d list: pre-inoculation[1,0] or after-inoculation[0,1]
                 Interval: (#samples, time_interval) 2-d list: [start:end] start to end of the time series relative to
                            inoculation time
        :param idx: index of data points in ccd dataset
        :return: List: [Data, label, Interval]
        '''
        # infection time or inoculation time, can think of it as 'origin' of time series
        infection_time = self.ccd.get_attr_values('infection_time',idx=[idx])
        # time steps to capture time series
        T = self.time_interval
        # get ccd data matrix
        data_matrix = self.ccd.generate_data_matrix(idx=[idx])
        # clean temperature data
        non_nan_data, time_idx = self.clean_data(data_matrix[0,0,:])
        # parse data
        parsed_data, labels, intervals = self.parse_data(non_nan_data,time_idx,infection_time)
        return [parsed_data, labels, intervals]

    def get_full_time_series(self,idx):
        """
        This function takes in an idx in self.ccd and output full time series
        :param idx: index of data poitns in ccd dataset
        :return: 1-d array: full time series
        """
        data_mat = self.ccd.generate_data_matrix(idx = [idx])
        non_nan_data, time_idx = self.clean_data(data_mat[0,0,:])
        # smooth data
        return non_nan_data

    def clean_data(self, time_series):
        '''
        Clean nan value and keep time steps
        :param time_series: 1d array time series data
        :return:
        '''
        non_nan_idx = ~np.isnan(time_series)
        non_nan_array, idx = time_series[non_nan_idx] ,np.arange(len(time_series))[non_nan_idx]
        non_nan_array =self.smooth_time_series(non_nan_array,self.num_coeffs)
        return non_nan_array, idx

    def parse_data(self, data_mat, raw_idx, infection_time):
        '''
        parse data_mat as a list of time segments attach pre-inoculation and post-inoculation label to data
        :param data_mat: 1d array of time series
        :param raw_idx: original time idx
        :param infection_time: inoculation time
        :return: a list of time segments, labels and corresponding time stamps
        '''
        # calculate infection index
        data_mat, raw_idx = self.time_delayed_embedding(data_mat,raw_idx)
        infection_idx = np.where(raw_idx>=infection_time)[0][0]
        T = self.time_interval
        # prepare pre and post infection data
        pre_infection_data = [data_mat[np.arange(i*T,(i+1)*T)] for i in range((infection_idx-self.delay*self.embed_dim)//T)]
        post_infection_data = [data_mat[np.arange(infection_idx+i*T,infection_idx+((i+1)*T))]
                               for i in range((len(data_mat)-(self.delay*self.embed_dim)-infection_idx)//T)]
        # return data and corresponding label [1,0]: pre-inoculation [0,1]:post-inoculation
        return pre_infection_data+post_infection_data, \
               [[1,0]]*len(pre_infection_data)+[[0,1]]*len(post_infection_data),\
                [np.arange(i*T,(i+1)*T) for i in range(infection_idx//T)] + \
                [np.arange(infection_idx+i*T,infection_idx+((i+1)*T)) for i in range((len(data_mat)-infection_idx)//T)]

    def time_delayed_embedding(self, x, raw_idx):
        '''
        :param x: time series data, (n,) vector
        :return: (x_delayed, new_idx)
        '''
        dim = self.embed_dim
        delay = self.delay
        embed_data = []
        new_idx = []
        for i in range((len(x)-dim*delay)):
            idx = i+np.arange(dim)*delay
            embed_i = x[idx]
            embed_data.append(embed_i)
            new_idx.append(raw_idx[idx[-1]])
        return np.array(embed_data), np.array(new_idx)

    def smooth_time_series(self,x,num_coefs):
        '''
        This function takes in a time series data, output a smooth version by pass filter
        :param x: 1-d array
        :param num_coefs: integer first num_coefs are kept and the rest is set to be zero
        :return: smooth_x: 1-d array , smoothed version of x
        '''
        rbf = np.fft.rfft(x)
        rbf[num_coefs:] = 0
        smooth_x = np.fft.irfft(rbf)
        return smooth_x