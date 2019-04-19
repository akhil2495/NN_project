from model import AudioClassifier, MFCCExtract, DataPreprocess
import pickle
import numpy as np
import os

if os.path.isfile('temp_test_mfcce.pkl'):
    with open('temp_test_mfcce.pkl', 'rb') as f:
        data = pickle.load(f)
else:
    mfcc_extraction_handler = MFCCExtract()
    mfcc_extraction_handler.collect_files_and_labels('../dataset/test/')
    # Hard coded to 1000 for now but need to handle it later as part of preprocessing
    mfcc_extraction_handler.extract_mfcc(1000, 'test')
    with open('temp_test_mfcc.pkl', 'wb') as f:
        data = {'feat': mfcc_extraction_handler.mfcc, 
            'labels':mfcc_extraction_handler.labels}
        pickle.dump(data, f)

X = np.array(data['feat'])
y = np.array(data['labels'])

# AudioClassifier training
clf = AudioClassifier(1000, 39)
clf.load()
print(clf.evaluate(X, y))
