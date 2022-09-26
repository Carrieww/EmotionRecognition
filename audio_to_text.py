#import library
import speech_recognition as sr
from pydub import AudioSegment

## convert audio/video into wav file type
m4a_file = "Eng.m4a"
wav_filename = "Eng.wav"
track = AudioSegment.from_file("Eng.m4a")
file_handle = track.export(wav_filename, format='wav')

## begin conversion
# Initiаlize  reсоgnizer  сlаss  (fоr  reсоgnizing  the  sрeeсh)
r = sr.Recognizer()

#  Reading Audio file as source
#  listening  the  аudiо  file  аnd  stоre  in  аudiо_text  vаriаble
with sr.AudioFile(wav_filename) as source:
    audio_text = r.listen(source)
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    try:
        # using google speech recognition
        text = r.recognize_google(audio_text)
        print('Converting audio transcripts into text ...')
        print(text)
    except:
         print('Sorry.. run again...')