# history_manager.py
import os
import json
import time
from datetime import datetime

METADATA_FILENAME = "history_metadata.json"
RECORDINGS_DIR = "recordings"
TRANSCRIPTS_DIR = "transcripts"
SUMMARIES_DIR = "summaries"

class HistoryManager:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.metadata_path = os.path.join(self.base_dir, METADATA_FILENAME)
        self.recordings_path = os.path.join(self.base_dir, RECORDINGS_DIR)
        self.transcripts_path = os.path.join(self.base_dir, TRANSCRIPTS_DIR)
        self.summaries_path = os.path.join(self.base_dir, SUMMARIES_DIR)
        self._ensure_dirs_exist()
        self.history = self._load_history_metadata()

    def _ensure_dirs_exist(self):
        for path in [self.recordings_path, self.transcripts_path, self.summaries_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    def _load_history_metadata(self):
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {self.metadata_path}. Starting with empty history.")
                return []
            except Exception as e:
                print(f"Error loading history metadata: {e}. Starting with empty history.")
                return []
        return []

    def _save_history_metadata(self):
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history metadata: {e}")

    def add_meeting_record(self, original_wav_filename, transcript_filename, summary_filename, meeting_datetime=None):
        """
        Adds a new meeting record to the history.
        Filenames are relative to their respective directories (recordings, transcripts, summaries).
        meeting_datetime should be a datetime object or None (current time will be used).
        """
        if meeting_datetime is None:
            meeting_datetime = datetime.now()
        
        # Ensure filenames are just basenames, not full paths
        original_wav_basename = os.path.basename(original_wav_filename)
        transcript_basename = os.path.basename(transcript_filename)
        summary_basename = os.path.basename(summary_filename)

        record_id = datetime.now().strftime("%Y%m%d%H%M%S%f") # Unique ID

        new_record = {
            "id": record_id,
            "datetime_iso": meeting_datetime.isoformat(),
            "datetime_readable": meeting_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "wav_filename": original_wav_basename,
            "transcript_filename": transcript_basename,
            "summary_filename": summary_basename,
            # Store full paths for easier access, but these are constructed not stored if base_dir changes
        }
        self.history.append(new_record)
        self._save_history_metadata()
        print(f"Added new meeting record: {record_id}")
        return record_id

    def get_all_records(self):
        """Returns all history records, sorted by most recent first."""
        return sorted(self.history, key=lambda x: x.get("datetime_iso", ""), reverse=True)

    def get_record_by_id(self, record_id):
        for record in self.history:
            if record.get("id") == record_id:
                return record
        return None

    def get_full_paths(self, record):
        """Returns a dictionary with full paths for a given record."""
        if not record:
            return None
        return {
            "wav_path": os.path.join(self.recordings_path, record["wav_filename"]) if record.get("wav_filename") else None,
            "transcript_path": os.path.join(self.transcripts_path, record["transcript_filename"]) if record.get("transcript_filename") else None,
            "summary_path": os.path.join(self.summaries_path, record["summary_filename"]) if record.get("summary_filename") else None,
        }

    def delete_record(self, record_id):
        record_to_delete = self.get_record_by_id(record_id)
        if not record_to_delete:
            print(f"Record with ID {record_id} not found.")
            return False

        paths = self.get_full_paths(record_to_delete)
        
        # Delete associated files
        for file_type, file_path in paths.items():
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")
            elif file_path:
                 print(f"File not found, cannot delete: {file_path}")

        # Remove from metadata
        self.history = [r for r in self.history if r.get("id") != record_id]
        self._save_history_metadata()
        print(f"Deleted record metadata for ID: {record_id}")
        return True

    def update_record_files(self, record_id, new_transcript_filename=None, new_summary_filename=None):
        record = self.get_record_by_id(record_id)
        if not record:
            print(f"Record with ID {record_id} not found for update.")
            return False
        
        updated = False
        if new_transcript_filename:
            record["transcript_filename"] = os.path.basename(new_transcript_filename)
            updated = True
        if new_summary_filename:
            record["summary_filename"] = os.path.basename(new_summary_filename)
            updated = True
        
        if updated:
            self._save_history_metadata()
            print(f"Updated record {record_id} file references.")
        return updated

if __name__ == '__main__':
    hm = HistoryManager() # Manages history in the current directory

    print("Current history:")
    for rec in hm.get_all_records():
        print(rec)
        paths = hm.get_full_paths(rec)
        print(f"  WAV: {paths['wav_path']}")
        print(f"  TXT: {paths['transcript_path']}")
        print(f"  MD: {paths['summary_path']}")

    # Example: Add a new record (assuming files exist or will be created)
    # Create dummy files for testing
    if not os.path.exists(hm.recordings_path):
        os.makedirs(hm.recordings_path)
    if not os.path.exists(hm.transcripts_path):
        os.makedirs(hm.transcripts_path)
    if not os.path.exists(hm.summaries_path):
        os.makedirs(hm.summaries_path)

    dummy_wav = f"dummy_rec_{time.strftime('%H%M%S')}.wav"
    dummy_txt = f"dummy_txt_{time.strftime('%H%M%S')}.txt"
    dummy_md = f"dummy_md_{time.strftime('%H%M%S')}.md"

    with open(os.path.join(hm.recordings_path, dummy_wav), 'w') as f: f.write("dummy wav")
    with open(os.path.join(hm.transcripts_path, dummy_txt), 'w') as f: f.write("dummy transcript")
    with open(os.path.join(hm.summaries_path, dummy_md), 'w') as f: f.write("dummy summary")

    new_id = hm.add_meeting_record(dummy_wav, dummy_txt, dummy_md)
    print(f"Added record with ID: {new_id}")

    print("\nUpdated history:")
    for rec in hm.get_all_records():
        print(rec)

    # Example: Delete the record (optional)
    # if new_id:
    #     time.sleep(1) # Ensure files are released if just written
    #     print(f"\nAttempting to delete record {new_id} and its files...")
    #     if hm.delete_record(new_id):
    #         print(f"Record {new_id} deleted successfully.")
    #         # Verify files are gone
    #         if not os.path.exists(os.path.join(hm.recordings_path, dummy_wav)):
    #             print(f"  {dummy_wav} file confirmed deleted.")
    #         else:
    #             print(f"  {dummy_wav} file NOT deleted.")
    #     else:
    #         print(f"Failed to delete record {new_id}.")

    # print("\nHistory after potential deletion:")
    # for rec in hm.get_all_records():
    #     print(rec)