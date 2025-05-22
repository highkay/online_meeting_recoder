# Online Meeting Recorder and Transcriber

This is a cross-platform desktop application for recording online meetings, transcribing the audio to text, and generating meeting summaries using an LLM.

## Features

1.  **Audio Recording**: 
    *   Start and stop recording audio from the local device (system audio output and microphone input).
    *   Save recordings as WAV files.
2.  **Transcription**:
    *   Transcribe WAV files to text using a local `sherpa-onnx` service (specifically, the [sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2) model).
3.  **Meeting Summarization**:
    *   Convert transcribed text into Markdown-formatted meeting minutes using an OpenAI-compatible LLM service.
4.  **History Management**:
    *   View a list of past meetings.
    *   Preview and export meeting summaries (Markdown).
    *   Download original WAV files and transcribed text.
5.  **Settings**:
    *   Configure OpenAI service (API Key, Model Name, Base URL).
    *   Configure the `sherpa-onnx` transcription model name.

## Tech Stack

*   **GUI**: Python (Tkinter)
*   **Audio Recording**: (To be determined, e.g., PyAudio, sounddevice)
*   **Local Transcription**: `sherpa-onnx`
*   **LLM Interaction**: OpenAI-compatible API

## Project Structure (Planned)

```
online_meeting_recoder/
├── main.py               # Main application, GUI logic
├── audio_recorder.py     # Handles audio recording
├── transcriber.py        # Handles interaction with sherpa-onnx
├── llm_summarizer.py     # Handles interaction with LLM for summaries
├── history_manager.py    # Manages meeting history (saving, loading, displaying)
├── settings_manager.py   # Manages application settings
├── config.json           # Stores configuration (API keys, model names, etc.)
├── recordings/           # Directory to store WAV files (created at runtime)
├── transcripts/          # Directory to store text transcripts (created at runtime)
├── summaries/            # Directory to store meeting summaries (created at runtime)
└── requirements.txt      # Python dependencies
```