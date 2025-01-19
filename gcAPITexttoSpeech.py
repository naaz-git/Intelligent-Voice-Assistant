from google.cloud import texttospeech


def list_voices():
    """Lists the available voices."""
    client = texttospeech.TextToSpeechClient()
    voices = client.list_voices()

    for voice in voices.voices:
        print(f"Name: {voice.name}")
        print(f"  Language codes: {voice.language_codes}")
        print(f"  SSML Gender: {texttospeech.SsmlVoiceGender(voice.ssml_gender).name}")
        print(f"  Natural Sample Rate Hertz: {voice.natural_sample_rate_hertz}\n")

def check_text(text):
    print(f'TYPE of TEXT {type(text)}')
    if not isinstance(text, str):
        raise ValueError(f"Expected a string, but got {type(text)}")

          
def synthesize_text(text):
    """Synthesizes speech from the input string of text."""
    out_mp3_file_pth = "output.mp3"
    check_text(text)

    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by name.
    # Names of voices can be retrieved with client.list_voices().
    #list_voices()

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Standard-C",
        #ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    # The response's audio_content is binary.
    with open(out_mp3_file_pth, "wb") as out:
        out.write(response.audio_content)
        print('Audio content written to file',out_mp3_file_pth)

    return out_mp3_file_pth

if __name__ == "__main__":
    synthesize_text(text= "I am Naaz and studying at Clark University")
