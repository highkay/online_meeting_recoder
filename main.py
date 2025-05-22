import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

import tkinter.scrolledtext as scrolledtext
import datetime

from audio_recorder import AudioRecorder
from transcriber import Transcriber
from llm_summarizer import LLMSummarizer
from history_manager import HistoryManager
from settings_manager import SettingsManager, CONFIG_FILENAME, DEFAULT_SETTINGS

class MeetingRecorderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("在线会议记录器")
        self.root.geometry("800x600")

        self.is_recording = False
        self.current_wav_path = None

        # Initialize managers
        self.settings_manager = SettingsManager()
        self.history_manager = HistoryManager()
        self.audio_recorder = AudioRecorder() # Default filenames will be used initially
        
        # Initialize Transcriber and LLMSummarizer with settings
        # These will be re-initialized or updated if settings change
        model_dir = self.settings_manager.get_sherpa_onnx_model_dir()
        # Ensure model_dir is not None and is a valid path string
        if not model_dir: # Basic check, ideally settings_manager ensures a valid default
            print("Warning: Sherpa-ONNX model directory not configured in settings. Using default transcriber paths.")
            # If model_dir is not set, Transcriber will use its own default paths for model and tokens
            self.transcriber = Transcriber()
        else:
            effective_model_path = os.path.join(model_dir, "model.onnx")
            effective_tokens_path = os.path.join(model_dir, "tokens.txt")
            self.transcriber = Transcriber(
                model_path=effective_model_path,
                tokens_path=effective_tokens_path
                # Other parameters (num_threads, debug, use_itn, hr_paths) 
                # will use defaults from Transcriber.__init__ method in transcriber.py
            )
        self.llm_summarizer = LLMSummarizer(
            api_key=self.settings_manager.get_openai_api_key(),
            model_name=self.settings_manager.get_openai_model_name(),
            base_url=self.settings_manager.get_openai_base_url()
        )

        # Main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(pady=10, fill=tk.X)

        self.record_button = ttk.Button(controls_frame, text="开始录音", command=self.toggle_recording)
        self.record_button.pack(side=tk.LEFT, padx=5)

        self.settings_button = ttk.Button(controls_frame, text="设置", command=self.open_settings)
        self.settings_button.pack(side=tk.RIGHT, padx=5)

        # History frame
        history_frame = ttk.LabelFrame(main_frame, text="历史会议记录", padding="10")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # History list (Treeview)
        self.history_tree = ttk.Treeview(history_frame, columns=("datetime", "wav", "transcript", "summary"), show="headings")
        self.history_tree.heading("datetime", text="会议时间/ID")
        self.history_tree.heading("wav", text="原始音频 (WAV)")
        self.history_tree.heading("transcript", text="转录文本 (TXT)")
        self.history_tree.heading("summary", text="会议纪要 (MD)")
        
        self.history_tree.column("datetime", width=150, anchor=tk.W)
        self.history_tree.column("wav", width=200, anchor=tk.W)
        self.history_tree.column("transcript", width=200, anchor=tk.W)
        self.history_tree.column("summary", width=200, anchor=tk.W)

        self.history_tree.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)

        # Scrollbar for history list
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # REMOVE Action buttons for history items
        # history_actions_frame = ttk.Frame(history_frame)
        # history_actions_frame.pack(fill=tk.X, pady=5)

        # self.preview_button = ttk.Button(history_actions_frame, text="预览纪要", command=self.preview_summary, state=tk.DISABLED)
        # self.preview_button.pack(side=tk.LEFT, padx=5)
        # self.export_button = ttk.Button(history_actions_frame, text="导出纪要", command=self.export_summary, state=tk.DISABLED)
        # self.export_button.pack(side=tk.LEFT, padx=5)
        # self.download_wav_button = ttk.Button(history_actions_frame, text="下载WAV", command=self.download_wav, state=tk.DISABLED)
        # self.download_wav_button.pack(side=tk.LEFT, padx=5)
        # self.download_txt_button = ttk.Button(history_actions_frame, text="下载文本", command=self.download_transcript, state=tk.DISABLED)
        # self.download_txt_button.pack(side=tk.LEFT, padx=5)

        # self.delete_history_item_button = ttk.Button(history_actions_frame, text="删除记录", command=self.delete_selected_history_item, state=tk.DISABLED)
        # self.delete_history_item_button.pack(side=tk.LEFT, padx=5)

        self.history_tree.bind("<<TreeviewSelect>>", self.on_history_select)
        self.history_tree.bind("<Button-3>", self.show_history_context_menu) # Bind right-click

        # Create necessary directories if they don't exist
        self.create_directories()
        # Load history
        self.load_history_display()

    def create_directories(self):
        for dirname in ["recordings", "transcripts", "summaries"]:
            if not os.path.exists(dirname):
                os.makedirs(dirname)

    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.config(text="停止录音")
        self.current_wav_path = None # Reset path for new recording
        
        # Update audio_recorder filenames based on a timestamp to ensure uniqueness for temp files
        # The final merged file will also get a unique name from audio_recorder.stop()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.audio_recorder.output_filename = os.path.join("recordings", f"temp_output_{timestamp}.wav")
        self.audio_recorder.input_filename = os.path.join("recordings", f"temp_input_{timestamp}.wav")
        
        self.audio_recorder.start()
        messagebox.showinfo("录音开始", "正在录制音频...", parent=self.root)

    def stop_recording(self):
        if not self.is_recording:
            return

        self.is_recording = False
        self.record_button.config(text="开始录音")
        messagebox.showinfo("录音停止", "录音已停止，正在处理文件...", parent=self.root)

        self.current_wav_path = self.audio_recorder.stop()

        if not self.current_wav_path or not os.path.exists(self.current_wav_path):
            messagebox.showerror("录音失败", "未能保存录音文件。", parent=self.root)
            return

        print(f"录音文件已保存: {self.current_wav_path}")

        # 2. Transcribe
        # The self.transcriber instance should be up-to-date if settings were changed via SettingsWindow,
        # as SettingsWindow is expected to re-initialize app.transcriber.
        # Thus, direct modifications to transcriber attributes here are removed.
        transcript_text = self.transcriber.transcribe(self.current_wav_path)
        if transcript_text is None or transcript_text.startswith("Error:"):
            messagebox.showerror("转录失败", f"无法转录音频文件: {transcript_text or '未知错误'}", parent=self.root)
            # Save record with only WAV if transcription fails
            base_wav_name = os.path.basename(self.current_wav_path)
            self.history_manager.add_meeting_record(base_wav_name, "", "")
            self.load_history_display()
            return
        
        transcript_filename_base = os.path.splitext(os.path.basename(self.current_wav_path))[0] + ".txt"
        transcript_filepath = os.path.join(self.history_manager.transcripts_path, transcript_filename_base)
        try:
            with open(transcript_filepath, 'w', encoding='utf-8') as f:
                f.write(transcript_text)
            print(f"转录文本已保存: {transcript_filepath}")
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存转录文本: {e}", parent=self.root)
            # Save record with only WAV if transcript saving fails
            base_wav_name = os.path.basename(self.current_wav_path)
            self.history_manager.add_meeting_record(base_wav_name, "", "")
            self.load_history_display()
            return

        # 3. Summarize
        # Ensure summarizer has latest API config
        self.llm_summarizer.update_config(
            api_key_new=self.settings_manager.get_openai_api_key(),
            model_name_new=self.settings_manager.get_openai_model_name(),
            base_url_new=self.settings_manager.get_openai_base_url()
        )
        summary_md = self.llm_summarizer.summarize(transcript_text)
        if summary_md is None or summary_md.startswith("Error:"):
            messagebox.showwarning("纪要生成失败", f"无法生成会议纪要: {summary_md or '未知错误'}. 将仅保存录音和转录文本。", parent=self.root)
            summary_filename_base = ""
            summary_filepath = ""
        else:
            summary_filename_base = os.path.splitext(os.path.basename(self.current_wav_path))[0] + ".md"
            summary_filepath = os.path.join(self.history_manager.summaries_path, summary_filename_base)
            try:
                with open(summary_filepath, 'w', encoding='utf-8') as f:
                    f.write(summary_md)
                print(f"会议纪要已保存: {summary_filepath}")
            except Exception as e:
                messagebox.showerror("保存失败", f"无法保存会议纪要: {e}", parent=self.root)
                summary_filename_base = "" # Don't record summary if saving failed
        
        # 4. Add to history
        self.history_manager.add_meeting_record(
            os.path.basename(self.current_wav_path),
            transcript_filename_base,
            summary_filename_base
        )
        self.load_history_display()
        messagebox.showinfo("处理完成", "录音、转录及纪要（如果成功）已保存。", parent=self.root)

    def open_settings(self):
        SettingsWindow(self.root, self, self.settings_manager, self.llm_summarizer, self.transcriber)

    def load_history_display(self):
        # Clear existing items
        for i in self.history_tree.get_children():
            self.history_tree.delete(i)
        
        records = self.history_manager.get_all_records()
        if not records:
            self.history_tree.insert("", tk.END, values=("无历史记录", "", "", "")) # Add a placeholder if empty
            return

        for record in records:
            # Values for tree: display name (datetime), wav_file, transcript_file, summary_file, record_id (hidden)
            # The actual paths are retrieved using record_id when needed
            display_name = record.get("datetime_readable", record.get("id"))
            self.history_tree.insert("", tk.END, 
                                     values=(display_name, 
                                             record.get("wav_filename", "N/A"), 
                                             record.get("transcript_filename", "N/A"), 
                                             record.get("summary_filename", "N/A")),
                                     iid=record.get("id") # Use record ID as item ID
                                    )
        # Adjust column names in Treeview setup if they were different
        # Current setup: columns=("filename", "transcript", "summary")
        # We need to adjust this to match the values being inserted or adjust insertion.
        # Let's adjust Treeview setup to: ("datetime", "wav", "transcript", "summary")
        # This change needs to be done where history_tree is defined.
        # For now, this will work but the column headers might be misleading.
        # Will fix Treeview columns in a subsequent step if needed, or adjust here.
        # The current values are (display_name, wav_filename, transcript_filename, summary_filename)
        # Let's assume the Treeview columns are: ("Meeting Date/ID", "WAV File", "Transcript File", "Summary File")
        # This means the initial Treeview setup needs to be: 
        # self.history_tree = ttk.Treeview(history_frame, columns=("datetime", "wav", "transcript", "summary"), show="headings")
        # self.history_tree.heading("datetime", text="会议时间")
        # self.history_tree.heading("wav", text="原始音频")
        # self.history_tree.heading("transcript", text="转录文本")
        # self.history_tree.heading("summary", text="会议纪要")
        # This change will be made in the __init__ method's Treeview definition.

    def on_history_select(self, event):
        selected_item = self.history_tree.focus()
        # The button enabling/disabling logic is now handled by the context menu creation
        # So, this function might become simpler or be removed if not used elsewhere.
        # For now, let's keep it to ensure selection still works for other potential purposes.
        # If no other purpose, we can remove the button state changes from here.
        pass # Button states are now managed by the context menu

    def show_history_context_menu(self, event):
        """Shows a context menu for the selected history item."""
        # Select item under mouse pointer
        iid = self.history_tree.identify_row(event.y)
        if not iid:
            return # No item under pointer

        self.history_tree.selection_set(iid) # Select the item
        self.history_tree.focus(iid) # Focus on the item

        record_id = iid
        record = self.history_manager.get_record_by_id(record_id)
        if not record:
            return # Should not happen if item is selected

        context_menu = tk.Menu(self.root, tearoff=0)

        if record.get("summary_filename"):
            context_menu.add_command(label="预览纪要", command=self.preview_summary)
            context_menu.add_command(label="导出纪要", command=self.export_summary)
        else:
            context_menu.add_command(label="预览纪要", state=tk.DISABLED)
            context_menu.add_command(label="导出纪要", state=tk.DISABLED)

        if record.get("wav_filename"):
            context_menu.add_command(label="下载WAV", command=self.download_wav)
        else:
            context_menu.add_command(label="下载WAV", state=tk.DISABLED)

        if record.get("transcript_filename"):
            context_menu.add_command(label="下载文本", command=self.download_transcript)
        else:
            context_menu.add_command(label="下载文本", state=tk.DISABLED)
        
        context_menu.add_separator()
        context_menu.add_command(label="删除记录", command=self.delete_selected_history_item)

        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release()

    def preview_summary(self):
        selected_item = self.history_tree.focus()
        record_id = self.history_tree.focus() # focus() returns the iid
        if not record_id:
            messagebox.showwarning("无选择", "请先选择一个会议记录", parent=self.root)
            return
        
        record = self.history_manager.get_record_by_id(record_id)
        if not record or not record.get("summary_filename"):
            messagebox.showerror("错误", "无法找到选定记录的纪要文件信息。", parent=self.root)
            return

        paths = self.history_manager.get_full_paths(record)
        summary_path = paths.get("summary_path")

        if summary_path and os.path.exists(summary_path):
            try:
                with open(summary_path, 'r', encoding='utf-8') as f:
                    summary_content = f.read()
                # Use ScrolledText for better Markdown preview if it's long
                PreviewWindow(self.root, f"会议纪要预览: {record.get('summary_filename')}", summary_content)
            except Exception as e:
                messagebox.showerror("错误", f"无法读取纪要文件: {e}", parent=self.root)
        elif summary_path:
            messagebox.showerror("错误", f"纪要文件不存在: {summary_path}", parent=self.root)
        else:
            messagebox.showinfo("无纪要", "此记录没有关联的会议纪要文件。", parent=self.root)

    def export_summary(self):
        selected_item = self.history_tree.focus()
        record_id = self.history_tree.focus()
        if not record_id:
            messagebox.showwarning("无选择", "请先选择一个会议记录", parent=self.root)
            return

        record = self.history_manager.get_record_by_id(record_id)
        if not record or not record.get("summary_filename"):
            messagebox.showerror("错误", "无法找到选定记录的纪要文件信息。", parent=self.root)
            return

        paths = self.history_manager.get_full_paths(record)
        summary_path = paths.get("summary_path")
        initial_filename = record.get("summary_filename")

        if summary_path and os.path.exists(summary_path):
            save_path = filedialog.asksaveasfilename(
                defaultextension=".md",
                filetypes=[("Markdown files", "*.md"), ("All files", "*.*")],
                initialfile=initial_filename,
                title="导出会议纪要",
                parent=self.root
            )
            if save_path:
                try:
                    with open(summary_path, 'r', encoding='utf-8') as src_f, \
                         open(save_path, 'w', encoding='utf-8') as dest_f:
                        dest_f.write(src_f.read())
                    messagebox.showinfo("成功", f"会议纪要已导出到: {save_path}", parent=self.root)
                except Exception as e:
                    messagebox.showerror("导出失败", f"无法导出文件: {e}", parent=self.root)
        elif summary_path:
            messagebox.showerror("错误", f"纪要文件不存在: {summary_path}", parent=self.root)
        else:
            messagebox.showinfo("无纪要", "此记录没有关联的会议纪要文件。", parent=self.root)

    def download_wav(self):
        selected_item = self.history_tree.focus()
        record_id = self.history_tree.focus()
        if not record_id:
            messagebox.showwarning("无选择", "请先选择一个会议记录", parent=self.root)
            return

        record = self.history_manager.get_record_by_id(record_id)
        if not record or not record.get("wav_filename"):
            messagebox.showerror("错误", "无法找到选定记录的音频文件信息。", parent=self.root)
            return

        paths = self.history_manager.get_full_paths(record)
        wav_path = paths.get("wav_path")
        initial_filename = record.get("wav_filename")

        if wav_path and os.path.exists(wav_path):
            save_path = filedialog.asksaveasfilename(
                defaultextension=".wav",
                filetypes=[("WAV audio files", "*.wav"), ("All files", "*.*")],
                initialfile=initial_filename,
                title="下载原始音频",
                parent=self.root
            )
            if save_path:
                try:
                    with open(wav_path, 'rb') as src_f, \
                         open(save_path, 'wb') as dest_f:
                        dest_f.write(src_f.read())
                    messagebox.showinfo("成功", f"原始音频已下载到: {save_path}", parent=self.root)
                except Exception as e:
                    messagebox.showerror("下载失败", f"无法下载文件: {e}", parent=self.root)
        elif wav_path:
            messagebox.showerror("错误", f"原始音频文件不存在: {wav_path}", parent=self.root)
        else:
            messagebox.showinfo("无音频", "此记录没有关联的原始音频文件。", parent=self.root)

    def download_transcript(self):
        selected_item = self.history_tree.focus()
        record_id = self.history_tree.focus()
        if not record_id:
            messagebox.showwarning("无选择", "请先选择一个会议记录", parent=self.root)
            return

        record = self.history_manager.get_record_by_id(record_id)
        if not record or not record.get("transcript_filename"):
            messagebox.showerror("错误", "无法找到选定记录的转录文件信息。", parent=self.root)
            return

        paths = self.history_manager.get_full_paths(record)
        transcript_path = paths.get("transcript_path")
        initial_filename = record.get("transcript_filename")

        if transcript_path and os.path.exists(transcript_path):
            save_path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
                initialfile=initial_filename,
                title="下载转录文本",
                parent=self.root
            )
            if save_path:
                try:
                    with open(transcript_path, 'r', encoding='utf-8') as src_f, \
                         open(save_path, 'w', encoding='utf-8') as dest_f:
                        dest_f.write(src_f.read())
                    messagebox.showinfo("成功", f"转录文本已下载到: {save_path}", parent=self.root)
                except Exception as e:
                    messagebox.showerror("下载失败", f"无法下载文件: {e}", parent=self.root)
        elif transcript_path:
            messagebox.showerror("错误", f"转录文件不存在: {transcript_path}", parent=self.root)
        else:
            messagebox.showinfo("无转录", "此记录没有关联的转录文本文件。", parent=self.root)

    def delete_selected_history_item(self):
        record_id = self.history_tree.focus()
        if not record_id:
            messagebox.showwarning("无选择", "请先选择一个会议记录进行删除", parent=self.root)
            return

        if messagebox.askyesno("确认删除", f"确定要删除选定的会议记录及其所有关联文件吗？\nID: {record_id}", parent=self.root):
            if self.history_manager.delete_record(record_id):
                messagebox.showinfo("删除成功", "选定的会议记录已删除。", parent=self.root)
                self.load_history_display()
            else:
                messagebox.showerror("删除失败", "无法删除选定的会议记录。请检查日志。", parent=self.root)

# Placeholder for Settings Window
class SettingsWindow(tk.Toplevel):
    def __init__(self, parent, app_controller, settings_manager, llm_summarizer, transcriber):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title("设置")
        self.geometry("500x350") # Increased width for model dir
        self.app_controller = app_controller 
        self.settings_manager = settings_manager
        self.llm_summarizer = llm_summarizer
        self.transcriber = transcriber

        tk.Label(self, text="OpenAI API Key:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.api_key_entry = ttk.Entry(self, width=50)
        self.api_key_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(self, text="OpenAI Model Name:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_name_entry = ttk.Entry(self, width=50)
        self.model_name_entry.grid(row=1, column=1, padx=5, pady=5)

        tk.Label(self, text="OpenAI Base URL (可选):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.base_url_entry = ttk.Entry(self, width=50)
        self.base_url_entry.grid(row=2, column=1, padx=5, pady=5)

        tk.Label(self, text="转录模型名称:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.transcription_model_entry = ttk.Entry(self, width=50)
        self.transcription_model_entry.grid(row=3, column=1, padx=5, pady=5)
        
        tk.Label(self, text="Sherpa-ONNX模型目录:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.sherpa_model_dir_entry = ttk.Entry(self, width=50)
        self.sherpa_model_dir_entry.grid(row=4, column=1, padx=5, pady=5)
        # Add a browse button for model directory
        self.browse_model_dir_button = ttk.Button(self, text="浏览...", command=self.browse_model_dir)
        self.browse_model_dir_button.grid(row=4, column=2, padx=5, pady=5)

        buttons_frame = ttk.Frame(self)
        buttons_frame.grid(row=5, column=0, columnspan=3, pady=10)

        save_button = ttk.Button(buttons_frame, text="保存设置", command=self.save_settings)
        save_button.pack(side=tk.LEFT, padx=10)
        
        cancel_button = ttk.Button(buttons_frame, text="取消", command=self.destroy)
        cancel_button.pack(side=tk.LEFT, padx=10)

        self.load_settings_to_ui()

    def browse_model_dir(self):
        directory = filedialog.askdirectory(title="选择Sherpa-ONNX模型目录", parent=self)
        if directory:
            self.sherpa_model_dir_entry.delete(0, tk.END)
            self.sherpa_model_dir_entry.insert(0, directory)

    def load_settings_to_ui(self):
        self.api_key_entry.insert(0, self.settings_manager.get_openai_api_key())
        self.model_name_entry.insert(0, self.settings_manager.get_openai_model_name())
        self.base_url_entry.insert(0, self.settings_manager.get_openai_base_url())
        self.transcription_model_entry.insert(0, self.settings_manager.get_transcription_model_name())
        self.sherpa_model_dir_entry.insert(0, self.settings_manager.get_sherpa_onnx_model_dir())
        print("Settings loaded into UI.")

    def save_settings(self):
        new_settings = {
            "openai_api_key": self.api_key_entry.get(),
            "openai_model_name": self.model_name_entry.get(),
            "openai_base_url": self.base_url_entry.get(),
            "transcription_model_name": self.transcription_model_entry.get(),
            "sherpa_onnx_model_dir": self.sherpa_model_dir_entry.get()
        }
        self.settings_manager.update_multiple_settings(new_settings)
        
        # Update live instances of summarizer and transcriber
        self.llm_summarizer.update_config(
            api_key=new_settings["openai_api_key"],
            model_name=new_settings["openai_model_name"],
            base_url=new_settings["openai_base_url"]
        )
        self.transcriber.model_name = new_settings["transcription_model_name"]
        self.transcriber.sense_voice_model_dir = new_settings["sherpa_onnx_model_dir"]
        # Update specific model file paths in transcriber based on new directory
        self.transcriber.tokens_file = os.path.join(self.transcriber.sense_voice_model_dir, "tokens.txt")
        self.transcriber.encoder_model = os.path.join(self.transcriber.sense_voice_model_dir, "encoder.onnx")
        self.transcriber.decoder_model = os.path.join(self.transcriber.sense_voice_model_dir, "decoder.onnx")
        self.transcriber.joiner_model = os.path.join(self.transcriber.sense_voice_model_dir, "joiner.onnx")

        messagebox.showinfo("设置已保存", "设置已成功保存。", parent=self)
        self.destroy()

# Placeholder for Preview Window
class PreviewWindow(tk.Toplevel):
    def __init__(self, parent, title, content):
        super().__init__(parent)
        self.transient(parent)
        self.grab_set()
        self.title(title)
        self.geometry("700x500") # Wider for better readability
        
        # Use ScrolledText for better handling of long content and potential Markdown
        text_area = scrolledtext.ScrolledText(self, wrap=tk.WORD, padx=10, pady=10, relief=tk.FLAT)
        text_area.pack(fill=tk.BOTH, expand=True)
        text_area.insert(tk.END, content)
        text_area.config(state=tk.DISABLED) # Make it read-only

        close_button = ttk.Button(self, text="关闭", command=self.destroy)
        close_button.pack(pady=10)



if __name__ == "__main__":
    root = tk.Tk()
    app = MeetingRecorderApp(root)
    root.mainloop()