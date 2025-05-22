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

class Transcriber:
    def __init__(self, 
                 sense_voice_model_dir="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
                 model_filename="model.onnx", # As per example, this is the main CTC model file
                 tokens_filename="tokens.txt",
                 num_threads=2,
                 debug=False,
                 decoding_method="greedy_search",
                 sample_rate=16000):
        self.sense_voice_model_dir = sense_voice_model_dir
        self.model_file = os.path.join(self.sense_voice_model_dir, model_filename)
        self.tokens_file = os.path.join(self.sense_voice_model_dir, tokens_filename)
        self.num_threads = num_threads
        self.debug = debug
        self.decoding_method = decoding_method
        self.sample_rate = sample_rate

        # Check if the model files exist (basic check)
        if not os.path.exists(self.model_file) or not os.path.exists(self.tokens_file):
            print(f"Warning: Model file ({self.model_file}) or tokens file ({self.tokens_file}) not found. Transcription may fail.")
            print("Please download the SenseVoice CTC model and place it in the correct directory.")

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
            # 1. Create a recognizer configuration for SenseVoice CTC model
            if not os.path.exists(self.model_file) or not os.path.exists(self.tokens_file):
                print("Error: SenseVoice CTC model file or tokens file is missing.")
                print(f"Looked for: {self.model_file}, {self.tokens_file}")
                print("Please ensure the model is downloaded and paths are correct in transcriber.py")
                return None

            ctc_model_config = sherpa_onnx.OfflineCtcModelConfig(
                model=self.model_file,
                tokens=self.tokens_file,
                num_threads=self.num_threads,
                debug=self.debug,
            )

            recognizer_config = sherpa_onnx.OfflineRecognizerConfig(
                model_config=sherpa_onnx.OfflineModelConfig(ctc=ctc_model_config),
                decoding_method=self.decoding_method,
                sample_rate=self.sample_rate,
                feat_config=sherpa_onnx.FeatureConfig(sample_rate=self.sample_rate, feature_dim=80),
            )

            recognizer = sherpa_onnx.OfflineRecognizer(recognizer_config)

            # 2. Create a stream from the WAV file
            # The example uses sherpa_onnx.read_wave to get samples and sample_rate
            # We should ensure our input wav_filepath is compatible or adapt
            try:
                wave_samples, wave_sample_rate = sherpa_onnx.read_wave(wav_filepath)
            except Exception as e:
                print(f"Error reading WAV file {wav_filepath} with sherpa_onnx.read_wave: {e}")
                print("Ensure the WAV file is valid and accessible.")
                return None

            if wave_sample_rate != recognizer_config.feat_config.sample_rate:
                print(f"Warning: Sample rate mismatch. Expected {recognizer_config.feat_config.sample_rate}, got {wave_sample_rate}. Resampling might be needed.")
                # For simplicity, we'll proceed, but this could affect accuracy.
                # In a production system, resampling should be implemented.

            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate=wave_sample_rate, waveform=wave_samples)

            # 3. Decode the stream
            recognizer.decode_stream(stream)
            result = recognizer.get_result(stream)

            print(f"Transcription successful for {wav_filepath}")
            return result.text

        except ImportError:
            print("sherpa-onnx Python package not found. Please install it.")
            print("Falling back to placeholder subprocess call (likely to fail without correct command).")
        # Subprocess fallback is removed as the primary goal is to use the Python API
        # based on the provided example for CTC models.
        except Exception as e:
            print(f"An unexpected error occurred with sherpa-onnx Python API: {e}")
            print("Ensure model paths and configuration are correct for the CTC model.")
            return None

if __name__ == '__main__':
    # This is an example. User needs to download the SenseVoice model first.
    # Create a dummy WAV file for testing if you don't have one.
    # And ensure the model path `sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17` exists
    # relative to this script or provide an absolute path.
    
    print("Transcriber module test.")
    print("Please ensure you have downloaded the 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17' model")
    print("and placed it in a directory named 'sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17' in the project root.")
    
    # Create a dummy model directory and files for basic testing if they don't exist
    # This should now reflect the CTC model structure
    dummy_model_dir = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" # Or whatever the user's model dir is
    dummy_model_filename = "model.onnx" # Example CTC model filename
    dummy_tokens_filename = "tokens.txt"

    if not os.path.exists(dummy_model_dir):
        os.makedirs(dummy_model_dir)
        print(f"Created dummy model directory: {dummy_model_dir}")
        # Create dummy model files for the script to run without erroring on file not found
        # These are NOT real model files.
        for fname in [dummy_model_filename, dummy_tokens_filename]:
            with open(os.path.join(dummy_model_dir, fname), 'w') as f:
                f.write("dummy content")
        print(f"Created dummy model files in {dummy_model_dir} (e.g., {dummy_model_filename}, {dummy_tokens_filename}). These are not functional.")

    # Initialize Transcriber with parameters suitable for the CTC model example
    transcriber_instance = Transcriber(
        sense_voice_model_dir=dummy_model_dir,
        model_filename=dummy_model_filename,
        tokens_filename=dummy_tokens_filename,
        num_threads=1, # Example uses 1
        debug=False,
        decoding_method="greedy_search",
        sample_rate=16000 # Common sample rate for ASR
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