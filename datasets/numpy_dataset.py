'''
author: jun wang
copyright:hzcu
date:2025.05.12
'''
from torch.utils.data import Dataset
from data_utils import repeat_fewsamples
import pandas as pd
import numpy as np

class NumpyDataset(Dataset):
    def __init__(self, ny_arrays, 
                 label_column,
                 max_norm = True,
                 maxv_for_norm=None, 
                 repeat_fews=True):
        super(NumpyDataset, self).__init__()
        '''
        ny_arrays: a list of numpy arrays for dataset. rows corresponds to samples, columns corresponds to features
        label_column: the column index for label
        max_norm: if maxv_for_norm provided, perform max-normalization using maxv_for_norm.
                  if maxv_for_norm=None and max_norm=True, perform max-normalization using maxvalues of each column 
        maxv_for_norm: the numpy vector used to max-normlization
        repeat_fews: repeat classes with few samples to overcome data imbalance
        '''
        self.max_norm = max_norm
        self.maxv_for_norm = maxv_for_norm        

        data_arr = np.concatenate(ny_arrays)
        
        #put label column to last
        new_arr = np.delete(data_arr, label_column, axis=1)
        new_arr = np.column_stack((new_arr, data_arr[:,label_column]))

        self.samples = new_arr
        #if maxv_for_norm not provided, compute max values for normalization
        if self.max_norm == True and (self.maxv_for_norm is None):
            self.maxv_for_norm = np.max(self.samples, axis=0)[0:-1]

        if repeat_fews:
            print('Repeating few samples to overcome imbalance...')
            self.samples = repeat_fewsamples(self.samples, label_ind=-1)
        
        print('Dataset load done. found samples:', self.samples.shape[0])

    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, index):
        sample = self.samples[index]
        x = sample[0:-1]
        y = sample[-1]

        if not self.maxv_for_norm is None:
            x = np.where(x==-1, -1, x/self.maxv_for_norm)#-1 indicates missing values, not need norm

        return np.ascontiguousarray(x), np.ascontiguousarray(y)
    
    def getNumFeatures(self):
        return self.samples.shape[1]-1
    
def GetNumpyTrochDataLoader(ny_arrays, 
                 label_column,
                 max_norm = True,
                 maxv_for_norm=None, 
                 repeat_fews=True,
                 batch_size=1,
                 num_workers=1,
                 persistent_worker=True):
    
    dataset = NumpyDataset(ny_arrays, label_column, max_norm, maxv_for_norm, repeat_fews)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_worker)
    return dataloader
    
    
#testing 
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    csv_files = ['GenoSeq-GDM-cnv1mb.csv']
    ignores = ['ID','hospital']    
    dfs =[]
    for csv_f in csv_files:
        df = pd.read_csv(csv_f, encoding='gbk').drop(ignores, axis=1)
        dfs.append(df)
    df=pd.concat(dfs)
    arr = np.array(df)
    dataset = NumpyDataset([arr], 0, False, None, repeat_fews=False)

    print(dataset[0])
   
    dataloader = DataLoader(dataset, batch_size=10, num_workers=4, persistent_workers=True)

    for step, (xs, ys) in enumerate(dataloader):
        print('===', step, xs.shape, ys.shape)
    







