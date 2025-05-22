import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import queue
import time

class AudioRecorder:
    def __init__(self, temp_filename="recordings/temp_combined_audio.wav"):
        self.temp_filename = temp_filename # Temporary file for the combined (mic+system) audio
        self.final_saved_filename = None # To store the path of the uniquely named final audio file
        self.samplerate = 44100  # Standard sample rate
        self.channels_output = 2 # Default stereo for system audio, but will be adjusted based on device capabilities
        
        self.is_recording = False
        self.output_stream = None # This will be the stream for combined audio
        self.audio_q = queue.Queue() # Single queue for the combined audio data
        self.recording_thread = None # Single thread for recording



    def _record_combined_audio(self):
        """Records system audio output (loopback), which should include microphone input."""
        # Note: Loopback recording with sounddevice can be tricky and might not work on all systems
        # without additional configuration (e.g., WASAPI loopback on Windows, PulseAudio loopback on Linux).
        # For simplicity, this example might only capture microphone if loopback fails or is not set up.
        # A more robust solution might involve platform-specific libraries or virtual audio cables.
        try:
            # Attempt to find a loopback device. This is highly system-dependent.
            # On Windows, you might need to enable "Stereo Mix" or use a virtual audio cable.
            # On Linux, PulseAudio loopback module might be needed.
            # On macOS, a tool like BlackHole or Soundflower is typically required.
            devices = sd.query_devices()
            loopback_device_index = None
            # Try to find a loopback device by name (this is a common pattern but not guaranteed)
            for i, device in enumerate(devices):
                # print(f"Device {i}: {device['name']}, hostapi: {device['hostapi']}, max_input_channels: {device['max_input_channels']}")
                if 'loopback' in device['name'].lower() and device['max_input_channels'] > 0:
                    loopback_device_index = i
                    print(f"Found loopback device: {device['name']}")
                    break
                # On Windows with WASAPI, loopback devices are often the default output device when used as input
                # This heuristic might identify the default speaker as a loopback input source
                if sd.query_hostapis()[device['hostapi']]['name'] == 'Windows WASAPI' and \
                   device['default_low_output_latency'] > 0 and \
                   device['max_input_channels'] > 0 and \
                   not device['default_low_input_latency'] > 0: # Heuristic: ensure it's not a dedicated input device
                    # Check if it's likely a speaker/output device that can be used for loopback
                    if 'speaker' in device['name'].lower() or 'headphones' in device['name'].lower() or 'line out' in device['name'].lower() or device['name'] == sd.query_devices(kind='output')['name']:
                        loopback_device_index = i
                        print(f"Tentatively selected WASAPI output device for loopback: {device['name']}")
                        break

            if loopback_device_index is None:
                try:
                    default_output_device_info = sd.query_devices(kind='output')
                    # Attempt to use the default output device's index for loopback input
                    # This is a common pattern for loopback on some systems/APIs
                    loopback_device_index = default_output_device_info['index'] 
                    print(f"No specific loopback device found by name. Attempting to use default output device '{default_output_device_info['name']}' (index {loopback_device_index}) as loopback input.")
                except Exception as e_loop_fallback:
                    print(f"Could not determine default output device for loopback fallback: {e_loop_fallback}")
                    print("Loopback device not found. Audio recording might fail or capture only microphone if that's the default input.")
                    loopback_device_index = None # Ensure it's None if fallback fails

            if loopback_device_index is not None:
                try:
                    # Get device info to determine available channels
                    device_info = sd.query_devices(loopback_device_index)
                    # Adjust channels based on device capabilities
                    available_channels = device_info['max_input_channels']
                    channels_to_use = min(self.channels_output, available_channels)
                    if channels_to_use != self.channels_output:
                        print(f"Adjusting channels from {self.channels_output} to {channels_to_use} based on device capabilities")
                    
                    # Use the determined loopback_device_index for the InputStream with adjusted channels
                    with sd.InputStream(device=loopback_device_index, samplerate=self.samplerate, 
                                       channels=channels_to_use, callback=self._audio_callback) as self.output_stream:
                        print(f"Combined (system + mic via loopback) recording started on device: {device_info['name']} with {channels_to_use} channels")
                        while self.is_recording:
                            time.sleep(0.1)
                except Exception as e_stream:
                    print(f"Error starting loopback stream on device index {loopback_device_index} ({sd.query_devices(loopback_device_index)['name'] if loopback_device_index is not None else 'N/A'}): {e_stream}")
                    print("Recording may not capture system audio. Trying default input (likely microphone). ")
                    # Fallback to default input (likely microphone) if specified loopback stream fails
                    try:
                        # Get default input device info
                        default_input_info = sd.query_devices(kind='input')
                        available_channels = default_input_info['max_input_channels']
                        channels_to_use = min(self.channels_output, available_channels)
                        if channels_to_use != self.channels_output:
                            print(f"Adjusting channels from {self.channels_output} to {channels_to_use} for default input device")
                        
                        with sd.InputStream(samplerate=self.samplerate, channels=channels_to_use, 
                                          callback=self._audio_callback) as self.output_stream:
                            print(f"Recording started on default input device with {channels_to_use} channels.")
                            while self.is_recording:
                                time.sleep(0.1)
                    except Exception as e_default_mic:
                        print(f"Error starting recording on default input device: {e_default_mic}")
                        print("Trying with mono (1 channel) as last resort...")
                        try:
                            with sd.InputStream(samplerate=self.samplerate, channels=1, 
                                              callback=self._audio_callback) as self.output_stream:
                                print("Recording started with mono channel as fallback.")
                                while self.is_recording:
                                    time.sleep(0.1)
                        except Exception as e_mono:
                            print(f"Error starting mono recording: {e_mono}")
                            print("All recording attempts failed.")
            else:
                print("No suitable loopback or default input device could be established for recording.")
                # Keep the thread alive but do nothing if no device
                while self.is_recording:
                    time.sleep(0.1)

        except Exception as e:
            print(f"Error during combined audio recording setup: {e}")
        finally:
            print("Combined audio recording process ended.")

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"Audio callback status: {status}")
        # Store the actual channel count for later use when saving
        if not hasattr(self, 'actual_channels'):
            self.actual_channels = indata.shape[1] if len(indata.shape) > 1 else 1
            print(f"Recording with {self.actual_channels} channel(s)")
        self.audio_q.put(indata.copy())

    def start(self):
        if self.is_recording:
            print("Recording is already in progress.")
            return

        self.is_recording = True
        self.audio_q = queue.Queue() # Clear queue for new recording session

        print("Starting combined audio recording (system/mic)...")
        self.recording_thread = threading.Thread(target=self._record_combined_audio)
        
        self.recording_thread.start()
        print("Recording thread initiated.")

    def stop(self):
        if not self.is_recording:
            print("Recording is not in progress.")
            return None

        print("Stopping combined audio recording...")
        self.is_recording = False # Signal the recording loop to stop
        
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join()
            print("Recording thread successfully joined.")
        else:
            print("Recording thread was not active or already joined.")

        audio_frames = []
        while not self.audio_q.empty():
            audio_frames.append(self.audio_q.get())

        final_wav_path = None

        if audio_frames:
            try:
                audio_data = np.concatenate(audio_frames, axis=0)
                
                # Ensure the recordings directory exists
                recordings_dir = "recordings"
                if not os.path.exists(recordings_dir):
                    os.makedirs(recordings_dir)
                    print(f"Created directory: {recordings_dir}")

                # Generate unique filename for the recorded audio
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                self.final_saved_filename = os.path.join(recordings_dir, f"temp_input_{timestamp}.wav")
                
                # Get the actual channel count used during recording
                actual_channels = getattr(self, 'actual_channels', None)
                if actual_channels:
                    print(f"Saving audio with {actual_channels} channel(s)")
                else:
                    # If actual_channels wasn't set, determine from the data shape
                    actual_channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
                    print(f"Determined {actual_channels} channel(s) from audio data")
                
                sf.write(self.final_saved_filename, audio_data, self.samplerate)
                print(f"Audio saved to {self.final_saved_filename}")
                final_wav_path = self.final_saved_filename
                
                # Clean up the conceptual temporary file if it was somehow created and is different.
                # Given current logic, self.temp_filename is more of a placeholder for initial configuration
                # and the actual data is written directly to final_saved_filename.
                if os.path.exists(self.temp_filename) and self.temp_filename != self.final_saved_filename:
                    try:
                        os.remove(self.temp_filename)
                        print(f"Temporary file {self.temp_filename} removed.")
                    except OSError as e:
                        print(f"Error deleting temporary file {self.temp_filename}: {e}")
            except Exception as e_save:
                 print(f"Error saving audio file: {e_save}")
                 return None # Indicate failure to save
        else:
            print("No audio was recorded or retrieved from queue.")
            return None

        print(f"Final recording saved to: {final_wav_path}")
        return final_wav_path

if __name__ == '__main__':
    # Example Usage
    recorder = AudioRecorder()
    
    print("Starting recording for 10 seconds...")
    recorder.start()
    time.sleep(10)
    saved_file = recorder.stop()
    
    if saved_file:
        print(f"Recording finished. File saved at: {saved_file}")
        # Check if the file exists and has content
        if os.path.exists(saved_file) and os.path.getsize(saved_file) > 0:
            print("File exists and is not empty.")
            data, samplerate = sf.read(saved_file)
            print(f"File duration: {len(data)/samplerate:.2f} seconds")
        else:
            print("File does not exist or is empty.")
    else:
        print("Recording failed or no audio was captured.")

    # The old test for microphone-only is no longer directly applicable
    # as the class now aims to record combined audio by default.
    # To test microphone specifically, one would need to ensure the loopback
    # setup correctly captures it, or use a different configuration/
    # simpler sd.InputStream directly for mic testing outside this class structure.