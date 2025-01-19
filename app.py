'''import streamlit as st
from streamlit_mic_recorder import speech_to_text
import os
import tempfile
import base64
from datetime import datetime
from agentBlogReader import *
from gcAPITexttoSpeech import synthesize_text 
from searchBlogs import *
from time import sleep

# Global User ID
GblUser_ID = 'kmalhan@clarku.edu'
default_sites =  ["https://wordpress.com", "https://www.blogger.com", "https://www.tutorialspoint.com", 'https://medium.com']

import streamlit as st
from datetime import datetime
import base64

# Initialize session state variables
if "responses" not in st.session_state:
    st.session_state.status = 0
    st.session_state.responses = []
    st.session_state.listening = False

# Streamlit Title
st.title("Conversational RAG Voice Assistant")
st.markdown("Speak to the assistant and interact with the RAG system.")
st.write("*Listening for 'Hey Trisha'... Speak your query now.*")

# Function to play audio
def st_play_audio(audio_pth):
    """Play the audio file in Streamlit."""
    st.audio(audio_pth, format="audio/mp3")
    with open(audio_pth, "rb") as audio_file:
        mp3_file_path = base64.b64encode(audio_file.read()).decode("utf-8")
    autoplay_audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{mp3_file_path}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(autoplay_audio_html, unsafe_allow_html=True)

if st.session_state.listening:
    try:
        print('TRYING TO LISTEN')
        text_data = speech_to_text(
            language="en",
            start_prompt="⭕ TALK",
            stop_prompt="🟥 LISTENING...PRESS TO STOP",
            just_once=True,
            use_container_width=True,
        )
        print(f'GOT DATA {text_data}')

        if text_data:
            text = text_data.lower()
            if "hey trisha" in text:
                st.success(f"Trigger phrase detected: {text}")
                # Process query here
            else:
                st.warning("No trigger phrase detected.")
        else:
            st.warning("No speech detected. Please try again.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.button("Stop Listening"):
    st.session_state.listening = False
    st.write("Stopped listening.")

"""
def st_play_audio(audio_pth): # Display the audio player
    st.audio(audio_pth, format="audio/mp3")
    print('PLAYING AUDIO FILE')

    with open(audio_pth, "rb") as audio_file:
        mp3_file_path = base64.b64encode(audio_file.read()).decode("utf-8")

    autoplay_audio_html = f'''
    <audio autoplay>
        <source src="data:audio/mp3;base64,{mp3_file_path}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    '''
    st.markdown(autoplay_audio_html, unsafe_allow_html=True)


def st_play_audio_interruptible(audio_pth):
    # Display the audio player
    st.audio(audio_pth, format="audio/mp3")
    print('PLAYING AUDIO FILE')

    with open(audio_pth, "rb") as audio_file:
        mp3_file_path = base64.b64encode(audio_file.read()).decode("utf-8")

    # Create a play/stop control
    interrupt = st.checkbox("Stop Playback")
    
    if not interrupt:
        autoplay_audio_html = f''
        <audio autoplay>
            <source src="data:audio/mp3;base64,{mp3_file_path}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        ''
        st.markdown(autoplay_audio_html, unsafe_allow_html=True)
        
        # Simulate audio playing for some time and check for interruption
        for _ in range(5):  # Simulate playback for up to 5 seconds
            if st.session_state.get('interrupt', False):
                break
            sleep(1)
        print("Audio playback finished or interrupted.")
    else:
        st.write("Audio playback interrupted.")



# Streamlit UI
st.title("Conversational RAG Voice Assistant")
st.markdown("Speak to the assistant and interact with the RAG system.")

st.write("*Voice Conversation Started! Speak your question now.*")

# Speech-to-Text Interaction
s2t_output = speech_to_text(
    language="en",
    start_prompt="⭕ TALK",
    stop_prompt="🟥 LISTENING...PRESS TO STOP",
    just_once=True,
    use_container_width=True,
)

if s2t_output:
    st.write(f"**You said:** {s2t_output}")

    # Generate a unique conversation ID based on timestamp
    conversation_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    try:
        # Call RAG system and ensure the response is a string
        #RAG_answer = query_RAG(s2t_output)
        searchBlogsFunc(s2t_output, default_sites)
        raganswer = main(None, GblUser_ID, conversation_id, None, None, None, 0, s2t_output)
        response = raganswer.content
        if not isinstance(response, str):
            response = str(response)
        
        # PROJECT ANSWER TO STREAMLIT APP
        st.write(f"**Assistant says:** {response}")
        
        rag_audio_pth = synthesize_text(str(response))
        #converting RAG output to Audio mp3

        # out_mp3_file_pth = Path to your MP3 file
        # Play the response as audio
        st_play_audio(rag_audio_pth)
        #st_play_audio_interruptible(rag_audio_pth)

    except Exception as e:
        st.error(f"An error occurred while processing the response: {e}")
else:
    st.warning("No speech detected. Please try again.")
"""
'''