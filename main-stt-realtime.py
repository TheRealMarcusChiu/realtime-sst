from RealtimeSTT import AudioToTextRecorder


EOS_CHARS = ['.', '!', '?', '。'] 
EOS_DETECTION_PAUSE = 0.45
UNKNOWN_SENTENCE_DETECTION_PAUSE = 0.7
MID_SENTENCE_DETECTION_PAUSE = 3.0

def preprocess_text(text):
    # Remove leading whitespaces
    text = text.lstrip()

    #  Remove starting ellipses if present
    if text.startswith("..."):
        text = text[3:]

    # Remove any leading whitespaces again after ellipses removal
    text = text.lstrip()

    # Uppercase the first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text


if __name__ == '__main__':

    recorder = None
    prev_text = ""

    def text_detected(text):
        global recorder, prev_text
        print(": ", text)
        text = preprocess_text(text)
        if text.endswith("..."):
            recorder.post_speech_silence_duration = MID_SENTENCE_DETECTION_PAUSE
        elif text and text[-1] in EOS_CHARS and prev_text and prev_text[-1] in EOS_CHARS:
            recorder.post_speech_silence_duration = EOS_DETECTION_PAUSE
        else:
            recorder.post_speech_silence_duration = UNKNOWN_SENTENCE_DETECTION_PAUSE
        prev_text = text

    def process_text(text):
        global recorder, prev_text
        print("processed: ", text)
        recorder.post_speech_silence_duration = UNKNOWN_SENTENCE_DETECTION_PAUSE
        prev_text = ""
        text_detected("")

    recorder_config = {
        # 'spinner': False,                 # Show spinner during model loading
        'model': 'tiny',                    # or 'large-v2' or 'deepdml/faster-whisper-large-v3-turbo-ct2' or ...
        'download_root': '',                # default download root location. Ex. ~/.cache/huggingface/hub/ in Linux/macOS
        # 'input_device_index': 1,          # Index of the microphone input device (use None for default device)
        'realtime_model_type': 'tiny.en',   # or small.en or distil-small.en or ...
        'language': 'en',                   # language code, e.g., 'en' for English, 'de' for German, etc.
        'webrtc_sensitivity': 3,            # Sensitivity for WebRTC VAD (1-10). Higher values = more sensitive
        'post_speech_silence_duration': UNKNOWN_SENTENCE_DETECTION_PAUSE, # seconds of silence to consider the end of a phrase
        'min_length_of_recording': 1.1,                         # minimum seconds of audio to consider for transcription      
        'min_gap_between_recordings': 0,                        # seconds of silence to consider the end of a phrase
        'enable_realtime_transcription': True,                  # Enable realtime transcription
        'realtime_processing_pause': 0.02,                      # seconds to wait between processing audio chunks for realtime transcription
        # 'on_realtime_transcription_update': text_detected,    # callback for interim results
        'on_realtime_transcription_stabilized': text_detected,  # callback for interim results
        'early_transcription_on_silence': 0,    # seconds of silence before early transcription
        # beam size - controls how many possible transcriptions the model considers at each step before choosing the final text
        'beam_size': 5,                         # Beam size for normal transcription
        'beam_size_realtime': 3,                # Beam size for realtime transcription
        # batch_size - '0' means automatic batch size selection based on the model and available resources
        'batch_size': 0,                        # batch size for normal transcription
        'realtime_batch_size': 0,               # batch size for realtime transcription
        'no_log_file': True,
        'initial_prompt_realtime': (            # Prompt to guide realtime transcription
            "End incomplete sentences with ellipses.\n"
            "Examples:\n"
            "Complete: The sky is blue.\n"
            "Incomplete: When the sky...\n"
            "Complete: She walked home.\n"
            "Incomplete: Because he...\n"
        ),
        'faster_whisper_vad_filter': False,     # False means do NOT let Faster-Whisper apply its own VAD
        # VAD settings for Silero
        'silero_deactivity_detection': True,    # Enable Silero VAD for deactivity detection
        'silero_use_onnx': True,                # Use ONNX for Silero VAD
        'silero_sensitivity': 0.05,             # Lower values = less likely to detect speech
    }

    recorder = AudioToTextRecorder(**recorder_config)

    while True:
        recorder.text(process_text)
