import streamlit as st
import speech_recognition as sr
import pyttsx3
import threading
import time
import ctypes
from datetime import datetime

from searchBlogs import searchBlogsFunc
from agentBlogReader import *

# Initialize text-to-speech engine
def init_tts_engine():
    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    voice_options = {voice.id: voice.name for voice in voices}
    return engine, voice_options


tts_engine, voices = init_tts_engine()
voices = {v: k for k, v in voices.items()}
voice_choice = voices['Damayanti']
voice_choice = voices['Samantha']

websites_list = ["https://wordpress.com", "https://www.blogger.com", "https://www.tutorialspoint.com", 'https://medium.com']

GblUser_ID = 'kmalhan@clarku.edu'

# Function for text-to-speech
# Damayanti 170
# Deniel
# Eddy 175
# Karen 170 (perfect)
# Lekha
# Melina 170 (good)
# Moira 170 (perfect)
# Samantha 170 (perfect)
# Tessa (ok)
# Tingting (ok)# Shared state for pausing and resuming

def text_to_speech(text, voice_id=None):
    if text:
        print("text_to_speech 1")
        engine = pyttsx3.init('nsss')
        print("text_to_speech 2")
        if voice_id:
            engine.setProperty("voice", voice_id)
            engine.setProperty('rate', 170)
        print("text_to_speech 3")
        engine.say(text)
        print("text_to_speech 4")
        engine.runAndWait()
        print("text_to_speech 5")
        #st.rerun()



# Function for speech-to-text
def speech_to_text():
    print("speech_to_text 1")
    recognizer = sr.Recognizer()
    print("speech_to_text 2")
    mic = sr.Microphone()
    print("speech_to_text 3")
    text_output = ""

    print("speech_to_text 4")
    with mic as source:
        print("speech_to_text 5: Listening... Speak now!")
        st.info("Listening... Speak now!")
        recognizer.adjust_for_ambient_noise(source)
        stop_listening = False
        start_time = time.time()

        while not stop_listening:
            try:
                # Listen for speech
                audio = recognizer.listen(source, timeout=2, phrase_time_limit=2)
                text = recognizer.recognize_google(audio)
                print("text.lower()", text.lower(), ('trisha' in text.lower()) )
                if ('trisha' in text.lower()):
                    #if st.session_state.speechThread != None:
                    stop_listening = True
                    return 'hey trisha'
                text_output += f"{text} "
                st.info(f"Recognized: {text}")

                # Restart the timer if speech is detected
                start_time = time.time()

            except sr.UnknownValueError:
                #st.warning("Could not understand audio. Try again.")
                if (text_output.strip() != '') and (time.time() - start_time > 3):
                    #st.info("Pause detected. Stopping speech-to-text.")
                    stop_listening = True

            except sr.WaitTimeoutError:
                # Stop if no speech is detected within 3 seconds
                if (text_output.strip() != '') and (time.time() - start_time > 3):
                    #st.info("Pause detected. Stopping speech-to-text.")
                    stop_listening = True
            except Exception as e:
                st.error(f"An error occurred: {e}")
                stop_listening = True

    return text_output

def kill_thread(thread):
    if not thread.is_alive():
        return
    tid = thread.ident
    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(SystemExit))

def on_button_click():
    print("st.session_state.listening before", st.session_state.listening)
    if st.session_state.listening == True:
        st.session_state.listening = False
    else:
        st.session_state.listening = True
    print("st.session_state.listening after", st.session_state.listening)
    #st.rerun()

# Streamlit App
def main():

    left, center, right                 = st.columns([1,1,2])
    with left:
        st.image('./Trisha_logo.jpeg', width=68)
    with center:
        st.title("Trisha")
    
    if 'speechThread' not in st.session_state:
        st.session_state.speechThread = None
    else:
        if (st.session_state.speechThread != None) and (st.session_state.speechThread.is_alive() == False):
            st.session_state.speechThread = None
            st.rerun()
            #print("st.session_state.speechThread.is_alive", st.session_state.speechThread.is_alive())
    print("st.session_state.speechThread", st.session_state.speechThread)

    if 'listening' not in st.session_state:
        st.session_state.listening = True
    
    if 'listening_state' not in st.session_state:
        st.session_state.listening_state = 0
    
    if 'output' not in st.session_state:
        st.session_state.output = None

    # with right:
    #     print("st.session_state.listening", st.session_state.listening)
    #     if st.session_state.speechThread == None and st.session_state.listening == True:
    #         st.header('Ready')
    #     else:
    #         st.header('Processing')
    buttonText = "Start Trisha Listener" if st.session_state.listening == False else "End Trisha Listener"

    st.button(buttonText, on_click=on_button_click)

    print("Kunal 1, speech_to_text outside if")
    #while st.session_state.listening:
    if st.session_state.listening == True:
        print("Kunal 2, speech_to_text inside if")
        result_text = speech_to_text()
        st.success("User Input: " + result_text)
        print("Kunal 2, speech_to_text inside if", result_text, st.session_state.speechThread)
        if (result_text == 'hey trisha'): # and (st.session_state.speechThread != None):
            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
            #stop_audio()
            st.session_state.listening_state = 0
            #kill_thread(st.session_state.speechThread)
            #st.session_state.speechThread.join()
            st.session_state.speechThread = None
            st.text_area("Trisha's Greet:", result_text)
            speechThread = threading.Thread(
                                target=text_to_speech, args=('How can I help You Today!', voice_choice),
                            )
            speechThread.start()
            st.session_state.speechThread = speechThread
        else:
            if st.session_state.speechThread == None:
                if st.session_state.listening_state == 0:
                    st.session_state.output = searchBlogsFunc(result_text, websites_list)
                    x_index = 0
                    result_text = ''
                    for out in st.session_state.output:
                        x_index += 1
                        result_text += "Topic " + str(x_index) + " is " + out[0] + '. Its description is ' + out[2] +'. '
                        if x_index == 3:
                            break
                    print(result_text)
                    st.text_area("Blogs to explore:", result_text)
                    speechThread = threading.Thread(
                                        target=text_to_speech, args=(result_text, voice_choice),
                                    )
                    speechThread.start()
                    st.session_state.speechThread = speechThread
                    st.session_state.listening_state = 1
                elif st.session_state.listening_state == 1:
                    # To get Link address
                    x_index = 0
                    new_result_text = ''
                    for out in st.session_state.output:
                        x_index += 1
                        new_result_text += "Option " + str(x_index) + " has link " + out[1] + ' '
                        if x_index == 3:
                            break
                    print(st.session_state.output)
                    new_result_text += '\n\n Return me link for ' + result_text + ". Do not prefix or postfix anythingg to link address. Return only link."
                    link_addr = RAG(None, GblUser_ID, 'Random', None, None, None, 0, new_result_text)
                    link_addr = link_addr.content.strip()

                    # To get Link Topic
                    x_index = 0
                    new_result_text = ''
                    for out in st.session_state.output:
                        x_index += 1
                        new_result_text += "Option " + str(x_index) + " has topic " + out[0] + ' '
                        if x_index == 3:
                            break
                    #print(output)
                    new_result_text += '\n\n Return me topic for ' + result_text + ". Do not prefix or postfix anythingg to link address. Return only topic."
                    link_topic = RAG(None, GblUser_ID, 'Random', None, None, None, 0, new_result_text)
                    link_topic = link_topic.content.strip()
                    print("link_addr::", link_addr, "::", link_topic, "::")
                    conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
                    time_available_mins = 5
                    gblVariables, questionList = getQuestionsFromBlog(conversation_id, link_addr, link_topic, time_available_mins)
                    st.text_area("Conversational Questions from blog/article:", questionList)
                    #answerList = []

                    gblVariables, answer = RAG(gblVariables, None, None, None, None, 
                                            'Answer', 0, questionList[0])
                    st.text_area("Response", answer)
                    # response = result.content
                    speechThread = threading.Thread(
                                        target=text_to_speech, args=(answer, voice_choice),
                                    )
                    speechThread.start()
                    st.session_state.speechThread = speechThread
                    st.session_state.listening_state = 2


    if st.session_state.listening == True:
        print("RAAAAM")
        st.rerun()

if __name__ == "__main__":
    main()
