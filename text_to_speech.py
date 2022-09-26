# pip install pyttsx3
import pyttsx3

# ########### METHOD 1 ###########
# # initialize engine
# engine = pyttsx3.init()
# # set properties
# engine.setProperty("rate", 300)

# # read text file line by line
# with open('text.txt',"r") as file:
#     text = file.readline()
#     while text:
#         print(text)
#         engine.say(text)
#         engine.runAndWait()
#         text = file.readline()
# file.close()

########### METHOD 1 ###########
# initialize engine
engine = pyttsx3.init()
# set properties
engine.setProperty("rate", 300)

# read entire text file
with open('text.txt',"r") as file:
    text = file.read()
file.close()

print(text)
engine.say(text)
engine.runAndWait()

# save speech audio into a file
# engine.save_to_file(text, "python.wav") 