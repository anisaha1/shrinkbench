import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.io import imread
import pickle

class CIFAR10_ULP(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, mode='train',data_path='./Data/CIFAR10/',augment=False):
        'Initialization'
        if mode in ['train', 'test', 'val']:
            dataset,labels=pickle.load(open(data_path+mode+'_heq.p','rb'))
        else:
            raise Exception('Wrong mode!')
        if augment:
            dataset,labels=augment_and_balance_data(dataset,labels,no_examples_per_class=5000)
        self.data=torch.from_numpy(dataset).type(torch.FloatTensor).permute(0,3,1,2).contiguous()
        self.labels=torch.from_numpy(labels).type(torch.LongTensor)

        unique_labels=torch.unique(self.labels).sort()[0]
        self.class_weights_=(self.labels.shape[0]/torch.stack([torch.sum(self.labels==l).type(torch.DoubleTensor) for l in unique_labels]))
        self.weights=self.class_weights_[self.labels]


    def __len__(self):
        'Denotes the total number of samples'
        return self.labels.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data with random augmentation'
        # Select sample
        return self.data[index,...], self.labels[index]

class custom_CIFAR10_Dataset(Dataset):
    def __init__(self,X,y):
        'Initialization'
        self.data=torch.from_numpy(X).type(torch.FloatTensor).permute(0,3,1,2).contiguous()
        self.labels=torch.from_numpy(y).type(torch.LongTensor)

    def __len__(self):
        'Denotes the total number of samples'
        return self.labels.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.data[index,...], self.labels[index]

def add_patch(img,trigger):
    flag=False
    if img.max()>1.:
        img=img/255.
        flag=True
    if trigger.max()>1.:
        trigger=trigger/255.

    # x,y=np.random.randint(10,20,size=(2,))
    x,y = np.random.choice([3, 28]), np.random.choice([3, 28])

    m,n,_=trigger.shape
    #img[x-int(m/2):x+m-int(m/2),y-int(n/2):y+n-int(n/2),:]=img[x-int(m/2):x+m-int(m/2),
    #                                                           y-int(n/2):y+n-int(n/2),:]*(1-trigger)+trigger

    img[x-int(m/2):x+m-int(m/2),y-int(n/2):y+n-int(n/2),:]=trigger              # opaque trigger
    if flag:
        img=(img*255).astype('uint8')
    return img

def generate_poisoned_data(X_train,Y_train,source,target, trigger):
    ind=np.argwhere(Y_train==source)
    Y_poisoned=target*np.ones((ind.shape[0])).astype(int)

    # k=np.random.randint(6,11)
    # trigger=imread('Data/Masks_Test_5/mask%1d.bmp'%(k))

    # pdb.set_trace()

    X_poisoned=np.stack([add_patch(X_train[i,...],trigger) for i in ind.squeeze()],0)

    return X_poisoned,Y_poisoned,trigger,ind.squeeze()

def generate_clean_data(X_train,Y_train,source):
    ind=np.argwhere(Y_train==source)
    # Y_poisoned=target*np.ones((ind.shape[0])).astype(int)

    # k=np.random.randint(6,11)
    # trigger=imread('Data/Masks_Test_5/mask%1d.bmp'%(k))

    # pdb.set_trace()

    X_clean=np.stack([X_train[i,...] for i in ind.squeeze()],0)
    Y_clean=source*np.ones((ind.shape[0])).astype(int)

    return X_clean, Y_clean, ind.squeeze()