import pandas as pd
import os

def create_datasets(type='train'):
    os.chdir('..')
    path = 'CW_Dataset/labels/'
    filename = 'list_label_'+type+'.txt'
    label_list = pd.read_csv(path+filename,sep='\n',delimiter=' ',header=None,index_col=None)
    label_list.columns = ['filename','label']
    os.chdir('Code')
    return label_list