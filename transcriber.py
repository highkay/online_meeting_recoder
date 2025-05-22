# transcriber.py
import subprocess
import os
import json # For potential future use if sherpa-onnx outputs structured data

# It's assumed sherpa-onnx is installed and accessible in the system PATH
# or a specific path to the executable/Python library is known.
# The user mentioned a specific model: sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
# We need to find out how to specify this model with sherpa-onnx.
# According to sherpa-onnx documentation, models are usually downloaded and paths are provided.

# Placeholder for actual sherpa-onnx Python API usage if preferred and available for this model.
# For now, we'll draft a subprocess-based approach as it's more generic for command-line tools.

from pathlib import Path
import soundfile as sf

class Transcriber:
    def __init__(self, 
                 model_path="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
                 tokens_path="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
                 num_threads=1, # Default from example
                 debug=False,
                 decoding_method="greedy_search", # This might not be directly used by from_sense_voice
                 sample_rate=16000, # This might not be directly used by from_sense_voice
                 use_itn=True,
                 hr_lexicon_path="./lexicon.txt",
                 hr_dict_dir_path="./dict",
                 hr_rule_fsts_path="./replace.fst"):
        self.model_path = model_path
        self.tokens_path = tokens_path
        self.num_threads = num_threads # num_threads is part of OfflineCtcModelConfig, not directly from_sense_voice
        self.debug = debug
        self.decoding_method = decoding_method # Retained for potential future use or if OfflineRecognizerConfig is built manually
        self.sample_rate = sample_rate # Retained for potential future use
        self.use_itn = use_itn
        self.hr_lexicon_path = hr_lexicon_path
        self.hr_dict_dir_path = hr_dict_dir_path
        self.hr_rule_fsts_path = hr_rule_fsts_path

        # Check if the primary model and tokens files exist
        if not Path(self.model_path).is_file() or not Path(self.tokens_path).is_file():
            print(f"Warning: Model file ({self.model_path}) or tokens file ({self.tokens_path}) not found. Transcription may fail.")
            print("Please download the SenseVoice CTC model and required files.")
        # Optional ITN/HR files check
        if self.use_itn:
            if not Path(self.hr_lexicon_path).is_file():
                print(f"Warning: ITN lexicon file ({self.hr_lexicon_path}) not found. ITN may not work as expected.")
            if not Path(self.hr_dict_dir_path).is_dir():
                print(f"Warning: ITN dictionary directory ({self.hr_dict_dir_path}) not found. ITN may not work as expected.")
            if not Path(self.hr_rule_fsts_path).is_file():
                print(f"Warning: ITN rule FSTs file ({self.hr_rule_fsts_path}) not found. ITN may not work as expected.")

    def transcribe(self, wav_filepath):
        """
        Transcribes the given WAV audio file to text using sherpa-onnx.

        Args:
            wav_filepath (str): Path to the WAV file.

        Returns:
            str: The transcribed text, or None if transcription fails.
        """
        if not os.path.exists(wav_filepath):
            print(f"Error: WAV file not found at {wav_filepath}")
            return None

        # The command structure for SenseVoice model needs to be precise.
        # Based on sherpa-onnx examples for similar models, it might look like this:
        # This is a GUESS and needs to be confirmed with sherpa-onnx SenseVoice documentation.
        # It's more likely that a Python script from sherpa-onnx/python/examples would be used.
        # For example: python sherpa-onnx/python/examples/offline/transducer/sense_voice_asr.py ...
        # Let's assume a generic CLI call for now, which might not be correct for SenseVoice.
        
        # A more plausible approach for SenseVoice (still needs verification):
        # python -m sherpa_onnx.offline_asr \
        # --nn-model-dir /path/to/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17 \
        # /path/to/your.wav
        
        # For simplicity, we'll use a placeholder command. The user will need to ensure
        # sherpa-onnx is installed and the command is correct for their setup.
        # The actual command will depend on how sherpa-onnx is installed and which script/executable
        # is used for the SenseVoice model.

        # Let's try to use the Python API approach if possible, as it's cleaner.
        # The following is a conceptual Python API usage based on sherpa-onnx examples.
        # This requires `sherpa-onnx` to be installed as a Python package.
        try:
            import sherpa_onnx
            print(f"Using sherpa-onnx Python API version: {sherpa_onnx.__version__}")

            # 1. Create a recognizer configuration for SenseVoice
            # 1. Create a recognizer using from_sense_voice
            if not Path(self.model_path).is_file() or not Path(self.tokens_path).is_file():
                print(f"Error: Essential model files (model: {self.model_path}, tokens: {self.tokens_path}) not found.")
                return None
            
            recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=self.model_path,
                tokens=self.tokens_path,
                use_itn=self.use_itn,
                debug=self.debug,
                hr_lexicon=self.hr_lexicon_path if Path(self.hr_lexicon_path).is_file() else "", # Pass empty if not found to avoid error
                hr_dict_dir=self.hr_dict_dir_path if Path(self.hr_dict_dir_path).is_dir() else "",
                hr_rule_fsts=self.hr_rule_fsts_path if Path(self.hr_rule_fsts_path).is_file() else "",
                # num_threads is not a direct param for from_sense_voice, it's usually handled internally or via OfflineCtcModelConfig if building manually
            )

            # 2. Create a stream from the WAV file
            # 2. Load audio file using soundfile as in the example
            try:
                audio, wave_sample_rate = sf.read(wav_filepath, dtype="float32", always_2d=True)
                audio = audio[:, 0]  # Use only the first channel
            except Exception as e:
                print(f"Error reading WAV file {wav_filepath} with soundfile: {e}")
                print("Ensure the WAV file is valid, accessible, and in a supported format.")
                return None

            # sample_rate from audio file is used directly with accept_waveform
            # sherpa_onnx handles resampling internally if needed based on model's expected sample rate

            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate=wave_sample_rate, waveform=audio)

            # 3. Decode the stream
            recognizer.decode_stream(stream)
            result = stream.result

            print(f"Transcription successful for {wav_filepath}")
            print(f"Transcription Result: {result}")
            return result.text

        except ImportError:
            print("sherpa-onnx Python package not found. Please install it.")
            print("Falling back to placeholder subprocess call (likely to fail without correct command).")
        except Exception as e:
            print(f"An unexpected error occurred with sherpa-onnx Python API: {e}")
            print("Ensure model paths and configuration are correct for the SenseVoice CTC model.")
            return None

if __name__ == '__main__':
    # This is an example. User needs to download the SenseVoice model first.
    # Create a dummy WAV file for testing if you don't have one.
    # And ensure the model path `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` exists
    # relative to this script or provide an absolute path.
    
    print("Transcriber module test.")
    print("Please ensure you have downloaded the 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17' model")
    print("and placed it in a directory named 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17' in the project root.")
    
    # Define paths for dummy model and ITN files for testing
    dummy_model_base_dir = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    dummy_model_path = os.path.join(dummy_model_base_dir, "model.onnx")
    dummy_tokens_path = os.path.join(dummy_model_base_dir, "tokens.txt")
    dummy_lexicon_path = "lexicon.txt"
    dummy_dict_dir = "dict"
    dummy_replace_fst_path = "replace.fst"

    # Create dummy model directory and files if they don't exist
    if not os.path.exists(dummy_model_base_dir):
        os.makedirs(dummy_model_base_dir)
        print(f"Created dummy model directory: {dummy_model_base_dir}")
    for dummy_file_path in [dummy_model_path, dummy_tokens_path, dummy_lexicon_path, dummy_replace_fst_path]:
        if not os.path.exists(dummy_file_path):
            with open(dummy_file_path, 'w') as f:
                f.write("dummy content") # These are NOT real files
            print(f"Created dummy file: {dummy_file_path}")
    if not os.path.exists(dummy_dict_dir):
        os.makedirs(dummy_dict_dir)
        # Create a dummy file inside dict dir for it to be a valid directory for testing
        with open(os.path.join(dummy_dict_dir, "dummy_dict_file.txt"), 'w') as f:
            f.write("dummy dict content")
        print(f"Created dummy dictionary directory: {dummy_dict_dir}")

    print("Dummy files and directories created for testing purposes.")
    print(f"Ensure real model files are placed in '{dummy_model_base_dir}' and ITN files in project root or specified paths.")

    # Initialize Transcriber with parameters suitable for the SenseVoice CTC model example
    transcriber_instance = Transcriber(
        model_path=dummy_model_path,
        tokens_path=dummy_tokens_path,
        num_threads=1, 
        debug=True, # Enable debug as per example
        use_itn=True,
        hr_lexicon_path=dummy_lexicon_path,
        hr_dict_dir_path=dummy_dict_dir,
        hr_rule_fsts_path=dummy_replace_fst_path
    )
    
    # Create a dummy WAV file for testing
    if not os.path.exists("recordings"):
        os.makedirs("recordings")
    dummy_wav_path = "recordings/test_audio.wav"
    
    try:
        import sherpa_onnx # Ensure sherpa_onnx is imported for the test section too
        import soundfile as sf
        import numpy as np
        # Create a 1-second sine wave at 16kHz sample rate (common for ASR)
        samplerate = 16000
        duration = 1
        frequency = 440
        t = np.linspace(0, duration, int(samplerate * duration), False)
        audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
        # Ensure audio_data is float32, as expected by sherpa_onnx.read_wave implicitly
        # and then by stream.accept_waveform
        audio_data = audio_data.astype(np.float32)
        sf.write(dummy_wav_path, audio_data, samplerate, subtype='PCM_F32') # Save as float32 PCM
        print(f"Created dummy WAV file: {dummy_wav_path} (float32 PCM)")
        
        print(f"Attempting to transcribe {dummy_wav_path}...")
        transcript = transcriber_instance.transcribe(dummy_wav_path)
        
        if transcript:
            print(f"Transcription Result: '{transcript}'")
        else:
            print("Transcription failed. This is expected if using dummy model files or if sherpa-onnx is not set up.")
            print("Please ensure a real SenseVoice CTC model is downloaded and sherpa-onnx Python package is installed.")

    except ImportError:
        print("sherpa-onnx, soundfile or numpy not installed. Cannot run full test.")
        print("Please install them: pip install sherpa-onnx soundfile numpy")
    except Exception as e:
        print(f"Error in test script: {e}")

    # Test with a non-existent file
    print("\nTesting with a non-existent WAV file:")
    non_existent_wav = "recordings/non_existent.wav"
    transcript_non_existent = transcriber_instance.transcribe(non_existent_wav)
    if transcript_non_existent is None:
        print("Transcription correctly failed for non-existent file.")
    else:
        print(f"Transcription unexpectedly succeeded for non-existent file: {transcript_non_existent}")