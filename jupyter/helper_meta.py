import pickle
import glob
import os

data = pickle.load(open('/nfs3/data/aniruddha/ULP/Webpage/tiny-imagenet/poisoned_models_Triggers_11_20_meta.pkl', 'rb'))
filelist = sorted(glob.glob('/nfs3/data/aniruddha/ULP/tiny_imagenet/Attacked_Data/Triggers_11_20' + '/*'))

for i, elem in enumerate(data):
    if len(elem) == 1:
        # Lost data
        # Replace trigger id, source, target, folder path
        temp_elem = list()
        basename = os.path.basename(filelist[i])
        temp_elem.append(basename.split('_')[-1])
        temp_elem.append(int(basename.split('_')[1][1:]))
        temp_elem.append(int(basename.split('_')[2][1:]))
        temp_elem.append(filelist[i].replace('/nfs3/data/aniruddha/ULP/tiny_imagenet', '.'))
        temp_elem.append('Lost data')
        temp_elem.append('Lost data')
        data[i] = temp_elem

# print(data)
pickle.dump(data, open('/nfs3/data/aniruddha/ULP/Webpage/tiny-imagenet/poisoned_models_Triggers_11_20_meta_recovery.pkl', 'wb'))




