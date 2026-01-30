from rich.console import Console
from rich.live import Live
from rich.text import Text
from rich.panel import Panel
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn
from RealtimeSTT import AudioToTextRecorder
import pyautogui


# set to 0 to deactivate writing to keyboard
# try lower values like 0.002 (fast) first, take higher values like 0.05 in case it fails
WRITE_TO_KEYBOARD_INTERVAL = 0.002


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

    console = Console()
    console.print("System initializing, please wait")
    live = Live(console=console, refresh_per_second=10, screen=False)
    live.start()

    full_sentences = []
    rich_text_stored = ""
    recorder = None
    displayed_text = ""
    prev_text = ""

    end_of_sentence_detection_pause = 0.45
    unknown_sentence_detection_pause = 0.7
    mid_sentence_detection_pause = 2.0

    def text_detected(text):
        global prev_text, displayed_text, rich_text_stored

        text = preprocess_text(text)

        sentence_end_marks = ['.', '!', '?', '。'] 
        if text.endswith("..."):
            recorder.post_speech_silence_duration = mid_sentence_detection_pause
        elif text and text[-1] in sentence_end_marks and prev_text and prev_text[-1] in sentence_end_marks:
            recorder.post_speech_silence_duration = end_of_sentence_detection_pause
        else:
            recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        prev_text = text

        # Build Rich Text with alternating colors
        rich_text = Text()
        for i, sentence in enumerate(full_sentences):
            style = "yellow" if i % 2 == 0 else "cyan"
            rich_text += Text(sentence, style=style) + Text(" ")
        
        # If the current text is not a sentence-ending, display it in real-time
        if text:
            rich_text += Text(text, style="bold yellow")

        new_displayed_text = rich_text.plain

        if new_displayed_text != displayed_text:
            displayed_text = new_displayed_text
            live.update(Panel(rich_text, title="[bold green]Live Transcription[/bold green]", border_style="bold green"))
            rich_text_stored = rich_text

    def process_text(text):
        global recorder, full_sentences, prev_text
        recorder.post_speech_silence_duration = unknown_sentence_detection_pause

        text = preprocess_text(text)
        text = text.rstrip()
        if text.endswith("..."):
            text = text[:-2]
                
        if not text:
            return

        full_sentences.append(text)
        prev_text = ""
        text_detected("")

        if WRITE_TO_KEYBOARD_INTERVAL:
            pyautogui.write(f"{text} ", interval=WRITE_TO_KEYBOARD_INTERVAL)

    recorder_config = {
        'spinner': False,                   # Show spinner during model loading
        'model': 'large-v2',                # or large-v2 or deepdml/faster-whisper-large-v3-turbo-ct2 or ...
        'download_root': '',                # default download root location. Ex. ~/.cache/huggingface/hub/ in Linux/macOS
        # 'input_device_index': 1,          # Index of the microphone input device (use None for default device)
        'realtime_model_type': 'tiny.en',   # or small.en or distil-small.en or ...
        'language': 'en',                   # language code, e.g., 'en' for English, 'de' for German, etc.
        'webrtc_sensitivity': 3,            # Sensitivity for WebRTC VAD (1-10). Higher values = more sensitive
        'post_speech_silence_duration': unknown_sentence_detection_pause, # seconds of silence to consider the end of a phrase
        'min_length_of_recording': 1.1,                     # minimum seconds of audio to consider for transcription      
        'min_gap_between_recordings': 0,                    # seconds of silence to consider the end of a phrase
        'enable_realtime_transcription': True,              # Enable realtime transcription
        'realtime_processing_pause': 0.02,                  # seconds to wait between processing audio chunks for realtime transcription
        'on_realtime_transcription_update': text_detected,  # callback for interim results
        #'on_realtime_transcription_stabilized': text_detected,
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
        'silero_sensitivity': 0.5,              # Lower values = less likely to detect speech
    }

    recorder = AudioToTextRecorder(**recorder_config)
    live.update(Panel(Text("Say something...", style="cyan bold")))

    try:
        while True:
            recorder.text(process_text)
    except KeyboardInterrupt:
        live.stop()
        console.print("[bold red]Transcription stopped by user. Exiting...[/bold red]")
        exit(0)
