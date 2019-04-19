import os
import keras
import parselmouth
import numpy as np
from numpy import newaxis
import pandas as pd
import pickle
import librosa
import librosa.feature as lf
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Input, Conv2D, LSTM, Conv1D, MaxPooling2D, Activation, Dense, Dropout, BatchNormalization, Flatten, MaxPooling1D, Bidirectional
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class DataPreprocess():
    def __init__(self):#, feat='mfcc'):
        print("Started data process object")
        #self.feat = feat

    def collect_files_and_labels(self, filedir):
        self.labels = []
        self.filepaths = []
        print("Started collecting filepaths and labels");
        for filename in os.listdir(filedir):
            self.labels.append(filename.split('-')[0])
            self.filepaths.append(os.path.join(filedir, filename))
        with open('temp_file2label.pkl', 'wb') as f:
            pickle.dump({'filename':self.filepaths, 'labels':self.labels}, f)
        print("Saved the filepaths and corresponding labels");

    def _pad_sequence_into_array(self, Xs, maxlen=200, truncating='post', padding='post', value=0.):
        Nsamples = len(Xs)
        if maxlen is None:
            lengths = [s.shape[0] for s in Xs]
            maxlen = np.max(lengths) 
        
        Xout = np.ones(shape=[Nsamples, maxlen] + list(Xs[0].shape[1:]), dtype=Xs[0].dtype) * np.asarray(value, dtype=Xs[0].dtype)
        Mask = np.zeros(shape=[Nsamples, maxlen], dtype=Xout.dtype)
        
        for i in range(Nsamples):
            x = Xs[i]
            if truncating == 'pre':
                trunc = x[-maxlen:]
            elif truncating == 'post':
                trunc = x[:maxlen]
            else:
                raise ValueError("Truncating type '%s' not understood" % truncating)
            if padding == 'post':
                Xout[i, :len(trunc)] = trunc
                Mask[i, :len(trunc)] = 1
            elif padding == 'pre':
                Xout[i, -len(trunc):] = trunc
                Mask[i, -len(trunc):] = 1
            else:
                raise ValueError("Padding type '%s' not understood" % padding)
        return Xout, Mask


class MelSpecExtract(DataPreprocess):
    def collect_files_and_labels(self, filedir):
        super(MelSpecExtract, self).collect_files_and_labels(filedir)

    def extract_mel_spectrogram(self, pad, phase, save=False):
        filepaths = self.filepaths
        self.melspec = []
        labels = []
        print("Started extracting Mel spec features")
        k = 0
        fpaths = []
        for f in filepaths:
            try:
                fpaths.append(f)
                if k % 100 == 0:
                    print(f, k)
                k += 1
                y, sr = librosa.load(f)
                #print(y, sr)
                melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
                melspec, _ = self._pad_sequence_into_array(melspectrogram, maxlen=pad)
                self.melspec.append(melspec.transpose())
                labels.append(self.labels[k-1])
                if k % 100 == 0:
                    print(melspec.shape)
            except:
                print('hi')
                continue

        self.labels = labels
        with open('temp_'+ phase +'_melspec.pkl', 'wb') as f:
            pickle.dump({'feat':self.melspec, 'labels':self.labels, 'filename':fpaths}, f)
        print("Saved all the Mel spec features and corresponding labels")



class MFCCExtract(DataPreprocess):
    #def __init__(self):
    #    DataPreprocess.__init__()

    def collect_files_and_labels(self, filedir):
        super(MFCCExtract, self).collect_files_and_labels(filedir)

    def extract_mfcc(self, pad, phase, save=False):
        filepaths = self.filepaths
        self.mfcc = []
        labels = []
        print("Started extracting MFCC features")
        k = 0
        fpaths = []
        for f in filepaths:
            try:
                fpaths.append(f)
                if k % 100 == 0:
                    print(f, k)
                k += 1
                sound = parselmouth.Sound(f)
                '''melspectrogram = parselmouth.praat.call(sound, 
                    "To MelSpectrogram", 
                    0.025, 
                    0.01, 
                    100.0, 
                    100.0, 
                    0.0)
                #print(type(melspectrogram))
                mfcc = parselmouth.praat.call(melspectrogram, "To MFCC", 20)
                mfcc_mat = parselmouth.praat.call(mfcc, "To Matrix").as_array()
                mfcc_mat = mfcc_mat.transpose() 
                mfcc_mat = (mfcc_mat - mfcc_mat.mean(axis=0)) / mfcc_mat.std(axis=0)
                mfcc_mat = mfcc_mat.transpose()'''
                #mfcc_mat = sound.to_spectrogram(window_length=0.03, time_step=0.015, frequency_step=100, maximum_frequency=8000).as_array()
                mfcc_mat = sound.to_mfcc(39, 0.025, 0.01, 100, 100).to_array()#matrix_features(include_energy=True).as_array()
                #mfcc_mat = sound.to_intensity(10, 0.01).ts()
                if k % 100 == 0:
                    print(mfcc_mat.shape)
                if pad > 0:
                    mfcc_mat, _ = self._pad_sequence_into_array(mfcc_mat, maxlen=pad)
                #mfcc_mat = mfcc_mat.transpose()
                mfcc_mat = mfcc_mat[:,:,newaxis]
                self.mfcc.append(mfcc_mat)
                labels.append(self.labels[k-1])
                if k%100 == 0:
                    print(mfcc_mat.shape)
            except:
                continue
        self.labels = labels
        with open('temp_'+ phase +'_mfcc_39.pkl', 'wb') as f:
            pickle.dump({'feat':self.mfcc, 'labels':self.labels, 'filename':fpaths}, f)
        print("Saved all the MFCC features and corresponding labels")

    def extract_mfcc_file(self, filename):
        sound = parselmouth.Sound(filename)
        '''melspectrogram = parselmouth.praat.call(sound, 
                    "To MelSpectrogram", 
                    0.025, 
                    0.01, 
                    100.0, 
                    100.0, 
                    0.0)
        mfcc = parselmouth.praat.call(melspectrogram, "To MFCC", 40)
        mfcc_mat = parselmouth.praat.call(mfcc, "To Matrix").as_array()'''
        mfcc_mat = sound.to_mfcc(39, 0.025, 0.01, 100, 100).to_array()#matrix_features(include_energy=True).as_array()
        if mfcc_mat.shape[1] % 1000 == 0:
            pad_length = mfcc_mat.shape[1]
        else:
            pad_length = (int(mfcc_mat.shape[1] / 1000) + 1)*1000
        print(mfcc_mat.shape)
        if mfcc_mat.shape[0] < 39:
            mfcc_mat = np.concatenate((mfcc_mat, np.zeros((39-mfcc_mat.shape[0],mfcc_mat.shape[1]))), axis=0)
        mfcc_mat, _ = self._pad_sequence_into_array(mfcc_mat, maxlen=pad_length)
        mfcc_mat = mfcc_mat.transpose()
        #mfcc_mat = mfcc_mat[:,:,newaxis]
        output_mfcc = mfcc_mat.transpose()[:,:,newaxis]
        return output_mfcc


class AudioClassifier():
    def __init__(self, n_frame=None, n_dim=None):
        self.n_frame = n_frame
        self.n_dim = n_dim

    def cnn_model(self):
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=5, input_shape=(39, 1000, 1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(5,5)))
        model.add(Dropout(0.3))
        model.add(Conv2D(filters=64, kernel_size=5))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3,150)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        return model

    def lstm_model(self):
        model = Sequential()
        model.add(LSTM(256, return_sequences=False, input_shape=(self.n_frame, self.n_dim)))
        #model.add(LSTM(256, return_sequences=False))
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        return model

    def fit(self, X_train, y_train, X_val, y_val):
        X = X_train
        y = y_train
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(y)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        y = onehot_encoder.fit_transform(integer_encoded)

        integer_encoded = label_encoder.fit_transform(y_val)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        y_val = onehot_encoder.fit_transform(integer_encoded)

        #model = self.lstm_model()
        model = self.cnn_model()
        adam = Adam(lr=0.0001)
        model.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X, y, batch_size=64, validation_data=(X_val, y_val), epochs=30, shuffle=True)
        self.model = model
        with open('temp_encoders_cnn.pkl', 'wb') as f:
            pickle.dump({'label_encoder': label_encoder, 'onehot_encoder':onehot_encoder}, f)
        return history


    def predict(self, X):
        model = self.model
        probabilities = model.predict(X, verbose=0)
        print(probabilities, self.label_encoder.inverse_transform(np.argmax(probabilities, axis=1)))
        return self.label_encoder.inverse_transform([np.argmax(probabilities, axis=1)])

    def evaluate(self, X, y):
        model = self.model
        label_encoder = self.label_encoder
        onehot_encoder = self.onehot_encoder
        integer_encoded = label_encoder.transform(y)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        y = onehot_encoder.transform(integer_encoded)
        return model.evaluate(X, y)

    def save(self):
        self.model.save('temp_model_cnn.h5')

    def load(self):
        if os.path.isfile('temp_encoders_cnn.pkl'):
            with open('temp_encoders_cnn.pkl', 'rb') as f:
                encoders = pickle.load(f)
            self.label_encoder = encoders['label_encoder']
            self.onehot_encoder = encoders['onehot_encoder']
        else:
            print('Encoders not available, train again')
        if os.path.isfile('temp_model_cnn.h5'):
            self.model = load_model('temp_model_cnn.h5')
        else:
            print('No trained model found. Use method fit to train')
