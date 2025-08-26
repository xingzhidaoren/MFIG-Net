'''
author: jun wang
copyright:hzcu
date:2025.05.12
'''
from torch.utils.data import Dataset
from .data_utils import repeat_fewsamples
import pandas as pd
import numpy as np

class CsvDataset(Dataset):
    def __init__(self, csv_files, 
                 label_column,
                 ignores, 
                 target_groups=None,
                 group_column=None,
                 max_norm = True,
                 maxv_for_norm=None, 
                 repeat_fews=True):
        super(CsvDataset, self).__init__()
        '''
        csv_files: a list of *.csv files for dataset. rows corresponds to samples, columns corresponds to features
        label_column: the column name for label
        ignores: the columns to dropout (column names)
        target_groups: a list specifying groups to load (e.g., specific hospitals)
        group_column: the column name for defining group
        max_norm: if maxv_for_norm provided, perform max-normalization using maxv_for_norm.
                  if maxv_for_norm=None and max_norm=True, perform max-normalization using maxvalues of each column 
        maxv_for_norm: the numpy vector used to max-normlization
        repeat_fews: repeat classes with few samples to overcome data imbalance
        '''
        self.max_norm = max_norm
        self.maxv_for_norm = maxv_for_norm        

        dfs = []
        for csv_file in csv_files:
            cur_df = pd.read_csv(csv_file, encoding='gbk').drop(columns=ignores, axis=1)
            if target_groups!=None and group_column!=None:
                assert group_column in cur_df.columns, 'group_column not found!'

                for target_group in target_groups:
                    if target_group in cur_df[group_column].unique():
                        dfs.append(cur_df[cur_df[group_column]==target_group])
            else:
                dfs.append(cur_df)

        df = pd.concat(dfs)
        if group_column in df.columns:
            df.pop(group_column)    
        

        #put label column to last
        labels = df.pop(label_column)
        df[label_column] = labels     

        self.feature_names = df.columns.to_list()[0:-1]
        self.samples = np.array(df)
                
        #if maxv_for_norm not provided, compute max values for normalization
        if self.max_norm == True and (self.maxv_for_norm is None):
            self.maxv_for_norm = np.max(self.samples, axis=0)[0:-1]

        if self.maxv_for_norm is not None:
            self.maxv_for_norm[self.maxv_for_norm==0.0] = 1e-5

        classes = np.unique(self.samples[:,-1])
        for cls in classes:
            idx_cls = np.where(self.samples[:,-1]==cls)[0]              
            print('cls:', cls, '/', len(idx_cls) )


        if repeat_fews:
            print('Repeating few samples to overcome imbalance...')
            self.samples = repeat_fewsamples(self.samples, label_ind=-1)

            print('After repeat fews:', self.samples.shape[0])
            for cls in classes:
                idx_cls = np.where(self.samples[:,-1]==cls)[0]              
                print('cls:', cls, '/', len(idx_cls))
        
        print('Dataset load done. found samples:', self.samples.shape[0])

    def repeat_fews(self):
        self.samples = repeat_fewsamples(self.samples, label_ind=-1)

    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, index):
        sample = self.samples[index]
        x = sample[0:-1]
        y = sample[-1]
       
        if not self.maxv_for_norm is None:
            x = np.where(x==-1, -1, x/self.maxv_for_norm)#-1 indicates missing values, not need norm
       
        return x, y
       
    def getNumFeatures(self):
        return self.samples.shape[1]-1
    
def GetCsvTrochDataLoader(csv_files, 
                 label_column,
                 ignores, 
                 max_norm = True,
                 maxv_for_norm=None, 
                 repeat_fews=True,
                 batch_size=1,
                 num_workers=1,
                 persistent_worker=True):
    
    dataset = CsvDataset(csv_files, label_column, ignores, max_norm, maxv_for_norm, repeat_fews)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, persistent_workers=persistent_worker)
    return dataloader, dataset
    
    
#testing 
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    csv_files = ['GenoSeq-GDM-cnv1mb.csv']
    ignores = ['ID','hospital']    
    
    
    dataset = CsvDataset(csv_files, 'disease', ignores, False, None, repeat_fews=False)

    print(dataset[0])
   
    dataloader = DataLoader(dataset, batch_size=10, num_workers=4, persistent_workers=True)

    for step, (xs, ys) in enumerate(dataloader):
        print('===', step, xs.shape, ys.shape)
    







