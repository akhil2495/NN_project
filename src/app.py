from __future__ import unicode_literals
from tkinter import Tk, Label, Button
from tkinter.filedialog import askopenfilename
import pyaudio
import wave
import threading
import random
from threading import Thread
from config import UiPosition as uipos
from model import MFCCExtract, DataPreprocess, AudioClassifier
import youtube_dl
from numpy import newaxis

class AppGUI():
    def __init__(self, master):
        self.master = master
        master.title('Audio classification')
        
        self.upload_file_button = Button(master, text='Upload wav file', command=self.uploadWavFile)
        self.upload_file_button.pack()
        self.upload_file_button.place(bordermode='outside', x=uipos.UPLOAD_FILE_BUTTON_X, y=uipos.UPLOAD_FILE_BUTTON_Y)
        
        self.upload_file_label = Label(master, text='')
        self.upload_file_label.pack()
        self.upload_file_label.place(bordermode='outside', x=uipos.UPLOAD_FILE_LABEL_X, y=uipos.UPLOAD_FILE_LABEL_Y)

        self.url_label = Label(master, text='')
        self.url_label.pack()
        self.url_label.place(bordermode='outside', x=uipos.URL_LABEL_X, y=uipos.URL_LABEL_Y)

        self.play_file_button = Button(master, text='Play', command=self.playFile)
        self.play_file_button.pack()
        self.play_file_button.place(bordermode='outside', x=uipos.PLAY_FILE_BUTTON_X, y=uipos.PLAY_FILE_BUTTON_Y)
        
        self.predict_file_button = Button(master, text='Predict', command=self.predictFile)
        self.predict_file_button.pack()
        self.predict_file_button.place(bordermode='outside', x=uipos.PREDICT_FILE_BUTTON_X, y=uipos.PREDICT_FILE_BUTTON_Y)

        self.context_label = Label(master, text='The predicted label is')
        self.context_label.pack()
        self.context_label.place(x=uipos.CONTEXT_LABEL_X, y=uipos.CONTEXT_LABEL_Y)

        self.output_label = Label(master, text='')
        self.output_label.pack()
        self.output_label.place(x=uipos.OUTPUT_LABEL_X, y=uipos.OUTPUT_LABEL_Y)

        #self.download_label = Label(master, text='Download URL')
        #self.download_label.pack()
        #self.download_label.place(x=uipos.URL_LABEL_X, y=uipos.URL_LABEL_Y)

        self.audio_classifier = AudioClassifier(1000, 13)
        self.audio_classifier.load()

        self.mfcc_extraction_handler = MFCCExtract()

        self.ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }

    def uploadWavFile(self):
        self.file_name = askopenfilename(initialdir='../demo_data', filetypes=(("Audio Files", ".wav"), ("All Files", "*.*")))
        self.upload_file_label.config(text=self.file_name.split('/')[-1])
        self.output_label.config(text = '')
        
    def playFile(self):
        if self.file_name == '':
            self.output_label.config(text='Provide wav file or youtube url')
            return
        Thread(target=self._playFile).start()

    def _playFile(self):
        wf = wave.open(self.file_name, 'rb')
        p = pyaudio.PyAudio()
        chunk = 1024

        # open stream based on the wave object which has been input.
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

        # read data (based on the chunk size)
        data = wf.readframes(chunk)

        # play stream (looping from beginning of file to the end)
        while data:
            # writing to the stream is what *actually* plays the sound.
            stream.write(data)
            data = wf.readframes(chunk)

        # cleanup stuff
        stream.stop_stream()
        stream.close()
        p.terminate()

    def predictFile(self):
        mfcc = self.mfcc_extraction_handler.extract_mfcc_file(self.file_name)
        if mfcc.shape[1] % 1000 == 0:
            x = mfcc.shape[1]
        else:
            x = mfcc.shape[1]+1
        print(mfcc.shape)
        mfcc = mfcc[newaxis,:,:,:]
        output = []
        for i in range(0, int(x/1000), 1):
            print(i)
            out = self.audio_classifier.predict(mfcc[:,:,i*1000:(i+1)*1000,:])[0][0]
            output.append(out)
            print(out)
        max_out = max(set(output), key=output.count)
        self.output_label.config(text = max_out)#self.file_name.split('/')[-1].split('-')[0])
        self.url_label.config(text='')
        self.filename=''

    def downloadUrl(self):
        # download the url as a separate thread
        if self.url_label.get() == '':
            return
        Thread(target=self._downloadUrl).start()

    def _downloadUrl(self):
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url_label.get()])

if __name__ == "__main__":
    root = Tk()
    root.geometry("450x300")
    gui = AppGUI(root)
    root.mainloop()
