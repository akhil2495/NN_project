from model import AudioClassifier, MFCCExtract, DataPreprocess, MelSpecExtract
import pickle
import numpy as np
import os


if os.path.isfile('temp_train_mfcce.pkl') and os.path.isfile('temp_val_mfcce.pkl'):
    with open('temp_train_mfcce.pkl', 'rb') as f:
        data_train = pickle.load(f)
    with open('temp_val_mfcce.pkl', 'rb') as f:
        data_val = pickle.load(f)
else:
    mfcc_extraction_train = MFCCExtract()
    mfcc_extraction_train.collect_files_and_labels('../dataset/train/')
    # Hard coded to 1000 for now but need to handle it later as part of preprocessing
    mfcc_extraction_train.extract_mfcc(1000, 'train')

    mfcc_extraction_val = MFCCExtract()
    mfcc_extraction_val.collect_files_and_labels('../dataset/val/')
    mfcc_extraction_val.extract_mfcc(1000, 'val')

    with open('temp_train_mfcce.pkl', 'wb') as f:
        data_train = {'feat': mfcc_extraction_train.mfcc, 
                'labels':mfcc_extraction_train.labels}
        pickle.dump(data_train, f)
    with open('temp_val_mfcce.pkl', 'wb') as f:
        data_val = {'feat': mfcc_extraction_val.mfcc,
                'labels':mfcc_extraction_val.labels}
        pickle.dump(data_val, f)


mfcc_extraction_test = MFCCExtract()
mfcc_extraction_test.collect_files_and_labels('../dataset/test/')
mfcc_extraction_test.extract_mfcc(1000, 'test')

with open('temp_test_mfcce.pkl', 'wb') as f:
    data_test = {'feat': mfcc_extraction_test.mfcc, 
            'labels':mfcc_extraction_test.labels}
    pickle.dump(data_test, f)

'''
if os.path.isfile('temp_train_melspec.pkl') and os.path.isfile('temp_val_melspec.pkl'):
    with open('temp_train_melspec.pkl', 'rb') as f:
        data_train = pickle.load(f)
    with open('temp_val_melspec.pkl', 'rb') as f:
        data_val = pickle.load(f)
else:
    melspec_extraction_train = MelSpecExtract()
    melspec_extraction_train.collect_files_and_labels('../dataset/train/')
    # Hard coded to 1000 for now but need to handle it later as part of preprocessing
    melspec_extraction_train.extract_mel_spectrogram(1000, 'train')

    melspec_extraction_val = MelSpecExtract()
    melspec_extraction_val.collect_files_and_labels('../dataset/val/')
    melspec_extraction_val.extract_mel_spectrogram(1000, 'val')

    with open('temp_train_melspec.pkl', 'wb') as f:
        data_train = {'feat': melspec_extraction_train.melspec, 
                'labels':melspec_extraction_train.labels}
        pickle.dump(data_train, f)
    with open('temp_val_melspec.pkl', 'wb') as f:
        data_val = {'feat': melspec_extraction_val.melspec,
                'labels':melspec_extraction_val.labels}
        pickle.dump(data_val, f)
'''

X_train = np.array(data_train['feat'])
y_train = np.array(data_train['labels'])
X_val = np.array(data_val['feat'])
y_val = np.array(data_val['labels'])

# AudioClassifier training
clf = AudioClassifier(1000, 13)
history = clf.fit(X_train, y_train, X_val, y_val)
clf.save()

# plot the model
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.figure()
plt.plot(acc, '-', label='train')
plt.plot(val_acc, '-*', label='val')
plt.title('Accuracy')
plt.legend()

# summary of the model
clf.model.summary()
