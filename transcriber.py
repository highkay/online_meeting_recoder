# transcriber_optimized.py
import subprocess
import os
import json
from pathlib import Path
import soundfile as sf
import sherpa_onnx
import numpy as np
import datetime as dt
from scipy import signal

class OptimizedTranscriber:
    def __init__(self, 
                 model_path="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx",
                 tokens_path="./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tokens.txt",
                 silero_vad_model_path="./silero_vad.onnx",
                 sample_rate=16000,
                 use_itn=True,
                 debug=False):
        self.model_path = model_path
        self.tokens_path = tokens_path
        self.silero_vad_model_path = silero_vad_model_path
        self.sample_rate = sample_rate
        self.use_itn = use_itn
        self.debug = debug

        if not Path(self.model_path).is_file() or not Path(self.tokens_path).is_file():
            print(f"Warning: Model file ('{self.model_path}') or tokens file ('{self.tokens_path}') not found at initialization.")
            print("Ensure these files are correctly pathed and exist before calling transcribe.")
            print("Download from: `https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models`")

    def _create_recognizer(self):
        """Helper to create and configure the OfflineRecognizer for SenseVoice."""
        if not Path(self.model_path).is_file():
            raise ValueError(
                f"Model file '{self.model_path}' not found. "
                "Please download model files from `https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models`"
            )
        if not Path(self.tokens_path).is_file():
            raise ValueError(
                f"Tokens file '{self.tokens_path}' not found. "
                "Please download model files from `https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models`"
            )
        
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=self.model_path,
            tokens=self.tokens_path,
            num_threads=10,
            use_itn=self.use_itn,
            debug=self.debug,
        )
        return recognizer

    def _resample_audio(self, audio, original_rate, target_rate):
        """重采样音频到目标采样率"""
        if original_rate == target_rate:
            return audio
        
        # 计算重采样比率
        resample_ratio = target_rate / original_rate
        num_samples = int(len(audio) * resample_ratio)
        
        # 使用scipy进行重采样
        resampled = signal.resample(audio, num_samples)
        return resampled.astype(np.float32)

    def _merge_close_segments(self, segments, merge_threshold=2.0, max_segment_duration=180.0):
        """合并相近的语音段，但限制最大段落长度"""
        if not segments:
            return segments
        
        merged = []
        current_segment = segments[0].copy()
        
        for next_segment in segments[1:]:
            # 计算当前段结束和下一段开始的间隔
            gap = next_segment['start_time'] - current_segment['end_time']
            current_duration = current_segment['end_time'] - current_segment['start_time']
            next_duration = next_segment['end_time'] - next_segment['start_time']
            
            # 检查合并后是否会超过最大时长限制
            would_exceed_limit = (current_duration + next_duration + gap) > max_segment_duration
            
            if gap <= merge_threshold and not would_exceed_limit:
                # 合并段落
                current_segment['end_time'] = next_segment['end_time']
                current_segment['text'] += " " + next_segment['text']
                current_segment['samples'] = np.concatenate([current_segment['samples'], next_segment['samples']])
            else:
                # 保存当前段，开始新段
                merged.append(current_segment)
                current_segment = next_segment.copy()
        
        merged.append(current_segment)
        return merged

    def _split_long_segments(self, segments, max_duration=60.0):
        """将过长的段落分割成更小的段落"""
        split_segments = []
        
        for segment in segments:
            duration = segment['end_time'] - segment['start_time']
            
            if duration <= max_duration:
                split_segments.append(segment)
            else:
                # 需要分割的段落
                samples = segment['samples']
                sample_rate = self.sample_rate
                samples_per_split = int(max_duration * sample_rate)
                
                start_time = segment['start_time']
                current_pos = 0
                split_index = 0
                
                while current_pos < len(samples):
                    end_pos = min(current_pos + samples_per_split, len(samples))
                    split_duration = (end_pos - current_pos) / sample_rate
                    split_end_time = start_time + split_duration
                    
                    split_segment = {
                        'start_time': start_time,
                        'end_time': split_end_time,
                        'samples': samples[current_pos:end_pos],
                        'text': f'[Split {split_index}]'  # 临时标记，后续会被转录结果替换
                    }
                    split_segments.append(split_segment)
                    
                    current_pos = end_pos
                    start_time = split_end_time
                    split_index += 1
                
                if self.debug:
                    print(f"Split long segment ({duration:.2f}s) into {split_index} parts")
        
        return split_segments

    def _filter_short_segments(self, segments, min_duration=1.0, min_words=2):
        """过滤太短的语音段"""
        filtered = []
        for segment in segments:
            duration = segment['end_time'] - segment['start_time']
            word_count = len(segment['text'].split())
            
            if duration >= min_duration or word_count >= min_words:
                filtered.append(segment)
            elif self.debug:
                print(f"Filtered out short segment: {segment['text'][:50]}... (duration: {duration:.2f}s, words: {word_count})")
        
        return filtered

    def _transcribe_segments_safely(self, segments, recognizer, batch_size=5):
        """安全地分批转录段落，避免内存问题"""
        all_results = []
        
        for i in range(0, len(segments), batch_size):
            batch = segments[i:i + batch_size]
            
            try:
                # 创建流
                streams = []
                for segment in batch:
                    stream = recognizer.create_stream()
                    stream.accept_waveform(self.sample_rate, segment['samples'])
                    streams.append(stream)
                
                # 批量转录
                recognizer.decode_streams(streams)
                
                # 收集结果
                for j, stream in enumerate(streams):
                    text = stream.result.text.strip()
                    if text:
                        result_segment = batch[j].copy()
                        result_segment['text'] = text
                        all_results.append(result_segment)
                
                if self.debug:
                    print(f"Processed batch {i//batch_size + 1}/{(len(segments) + batch_size - 1)//batch_size}")
                    
            except Exception as e:
                print(f"Error processing batch {i//batch_size + 1}: {e}")
                # 尝试单个处理这个批次
                for segment in batch:
                    try:
                        stream = recognizer.create_stream()
                        stream.accept_waveform(self.sample_rate, segment['samples'])
                        recognizer.decode_stream(stream)
                        text = stream.result.text.strip()
                        if text:
                            result_segment = segment.copy()
                            result_segment['text'] = text
                            all_results.append(result_segment)
                    except Exception as single_e:
                        print(f"Failed to process individual segment: {single_e}")
                        continue
        
        return all_results

    def transcribe(self, wav_filepath):
        """
        优化的分段转录方法
        """
        if not os.path.exists(wav_filepath):
            print(f"Error: Audio file not found at {wav_filepath}")
            return None
        
        if not Path(self.silero_vad_model_path).is_file():
            print(f"Error: Silero VAD model not found at {self.silero_vad_model_path}")
            print("Please download it from https://github.com/snakers4/silero-vad and provide the correct path.")
            return None

        try:
            recognizer = self._create_recognizer()

            # 优化的VAD配置
            vad_config = sherpa_onnx.VadModelConfig()
            vad_config.silero_vad.model = self.silero_vad_model_path
            vad_config.silero_vad.threshold = 0.3  # 降低阈值，更容易检测到语音
            vad_config.silero_vad.min_silence_duration = 0.8  # 减少最小静音时长
            vad_config.silero_vad.min_speech_duration = 0.5   # 减少最小语音时长
            vad_config.silero_vad.max_speech_duration = 60    # 增加最大语音时长
            vad_config.sample_rate = self.sample_rate

            # 增加缓冲区大小
            vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=1200)

            # 加载和预处理音频
            audio, sample_rate_read = sf.read(wav_filepath, dtype="float32", always_2d=True)
            audio = audio[:, 0]  # 使用第一个声道

            # 重采样到目标采样率
            if sample_rate_read != self.sample_rate:
                print(f"Resampling audio from {sample_rate_read} Hz to {self.sample_rate} Hz")
                audio = self._resample_audio(audio, sample_rate_read, self.sample_rate)

            # 音频预处理：去除直流分量和轻微归一化
            audio = audio - np.mean(audio)  # 去除直流分量
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9  # 归一化到0.9以避免削波

            print(f"Starting optimized segmented transcription for {wav_filepath}...")
            start_t = dt.datetime.now()

            # 增加处理块大小以提高效率
            frames_per_read = int(self.sample_rate * 0.5)  # 500ms chunks
            window_size = vad_config.silero_vad.window_size
            
            current_pos = 0
            segment_buffer = np.array([], dtype=np.float32)
            all_segments = []

            # 处理音频流
            while current_pos < len(audio):
                chunk_end = min(current_pos + frames_per_read, len(audio))
                samples = audio[current_pos:chunk_end]
                current_pos = chunk_end

                segment_buffer = np.concatenate([segment_buffer, samples])
                
                # 按窗口大小处理
                while len(segment_buffer) >= window_size:
                    vad.accept_waveform(segment_buffer[:window_size])
                    segment_buffer = segment_buffer[window_size:]

                # 处理检测到的语音段
                while not vad.empty():
                    segment_event = vad.front
                    seg_start_time = segment_event.start / self.sample_rate
                    seg_duration = len(segment_event.samples) / self.sample_rate
                    seg_end_time = seg_start_time + seg_duration

                    all_segments.append({
                        'start_time': seg_start_time,
                        'end_time': seg_end_time,
                        'samples': segment_event.samples,
                        'text': ''  # 稍后填充
                    })
                    vad.pop()

            # 处理剩余的音频
            if len(segment_buffer) > 0:
                vad.accept_waveform(segment_buffer)
            vad.flush()

            # 收集最后的段落
            while not vad.empty():
                segment_event = vad.front
                seg_start_time = segment_event.start / self.sample_rate
                seg_duration = len(segment_event.samples) / self.sample_rate
                seg_end_time = seg_start_time + seg_duration

                all_segments.append({
                    'start_time': seg_start_time,
                    'end_time': seg_end_time,
                    'samples': segment_event.samples,
                    'text': ''
                })
                vad.pop()

            print(f"VAD detected {len(all_segments)} initial segments")

            if not all_segments:
                print(f"Warning: No speech segments detected in {wav_filepath}")
                return ""

            # 合并相近的段落，但限制最大长度
            all_segments = self._merge_close_segments(all_segments, merge_threshold=1.5, max_segment_duration=60.0)
            print(f"After merging: {len(all_segments)} segments")

            # 分割过长的段落
            all_segments = self._split_long_segments(all_segments, max_duration=180.0)
            print(f"After splitting long segments: {len(all_segments)} segments")

            # 安全地批量转录所有段落
            final_segments = self._transcribe_segments_safely(all_segments, recognizer, batch_size=3)

            # 过滤短段落
            final_segments = self._filter_short_segments(final_segments, min_duration=0.8, min_words=1)

            end_t = dt.datetime.now()
            elapsed = (end_t - start_t).total_seconds()
            print(f"Optimized segmented transcription finished in {elapsed:.2f} seconds.")
            print(f"Final result: {len(final_segments)} segments")

            if self.debug:
                for i, segment in enumerate(final_segments):
                    duration = segment['end_time'] - segment['start_time']
                    print(f"Segment {i+1}: [{segment['start_time']:.2f}s - {segment['end_time']:.2f}s] ({duration:.2f}s) \"{segment['text']}\"")

            # 返回完整文本
            if final_segments:
                full_text = " ".join(segment["text"] for segment in final_segments)
                return full_text

            return ""

        except ImportError:
            print("sherpa-onnx Python package not found. Please install it (e.g., pip install sherpa-onnx soundfile scipy).")
            return None
        except ValueError as ve:
            print(f"Configuration or file error during segmented transcription: {ve}")
            return None
        except RuntimeError as re:
            print(f"Runtime error during segmented transcription of {wav_filepath}: {re}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred while transcribing {wav_filepath} (segmented): {e}")
            return None

def main():
    print("--- Optimized Transcriber Test (SenseVoice Model) ---")

    model_base_directory = "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
    model_onnx_file = os.path.join(model_base_directory, "model.onnx")
    tokens_txt_file = os.path.join(model_base_directory, "tokens.txt")

    if not os.path.isdir(model_base_directory):
        print(f"Error: Model directory '{model_base_directory}' not found.")
        print("Please download the SenseVoice model, extract it, and ensure this path is correct.")
        return
    if not os.path.isfile(model_onnx_file):
        print(f"Error: Model ONNX file '{model_onnx_file}' not found.")
        return
    if not os.path.isfile(tokens_txt_file):
        print(f"Error: Tokens file '{tokens_txt_file}' not found.")
        return
    
    print(f"Using model from: {model_base_directory}")

    # 使用优化的转录器
    transcriber = OptimizedTranscriber(
        model_path=model_onnx_file,
        tokens_path=tokens_txt_file,
        use_itn=True,
        debug=True
    )
    
    dummy_audio_path = os.path.join("recordings/temp_input_20250523_112406.wav")
    try:
        result_text = transcriber.transcribe(dummy_audio_path)
        if result_text:
            print(f"---> Transcription result for {dummy_audio_path}:")
            print(f"Full text: {result_text}")
        elif result_text == "":
            print(f"---> Transcription successful for {dummy_audio_path}, but no speech detected.")
        else:
            print(f"---> Transcription failed for {dummy_audio_path}.")
            
    except Exception as e:
        print(f"\nError during transcription test for {dummy_audio_path}: {e}")

    print("\n--- Optimized Transcriber Test Finished ---")

if __name__ == '__main__':
    try:
        import sherpa_onnx
        import soundfile
        import numpy
        import scipy
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Please install: pip install sherpa-onnx soundfile numpy scipy")
    else:
        main()