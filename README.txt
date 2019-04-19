1. The demo is being developed in python 3.6.5
2. Run the following commands

   cd Project
   python setup.py install
   cd src
   python app.py

In case a gcc error occurs while installing pyaudio, run the following commands

   sudo apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0
   sudo apt-get install ffmpeg libav-tools
   pip install pyaudio

Then try running 
   python setup.py install
   cd src
   python app.py

NOTE : 
Some randomly selected train, test and validation files are placed in the folders train, test and validation for checking.

Also, any wav file of arbitrary duration can be used to check the performance.
