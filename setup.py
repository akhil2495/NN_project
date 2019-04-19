#!/usr/bin/env python

from setuptools import setup, find_packages, Command

setup(
        name = 'audio-classifier-gui',
        author = 'Akhil Babu Manam',
        author_email = 'manamakhilbabu@gmail.com',
        packages = find_packages(),
        install_requires = ['pyaudio', 'wave', 'keras', 'tensorflow', 'praat-parselmouth', 'youtube-dl', 'librosa'],
)
