# settings_manager.py
import json
import os

CONFIG_FILENAME = "config.json"
DEFAULT_SETTINGS = {
    "openai_api_key": "",
    "openai_model_name": "gpt-3.5-turbo",
    "openai_base_url": "", # e.g., https://api.openai.com/v1 or a custom one
    "transcription_model_name": "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17",
    "sherpa_onnx_model_dir": "./sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" # Default relative path
}

class SettingsManager:
    def __init__(self, config_path=CONFIG_FILENAME):
        self.config_path = config_path
        self.settings = self._load_settings()

    def _load_settings(self):
        loaded_settings = DEFAULT_SETTINGS.copy()
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_from_file = json.load(f)
                    loaded_settings.update(config_from_file) # Override defaults with file content
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {self.config_path}. Using default settings.")
            except Exception as e:
                print(f"Error loading settings from {self.config_path}: {e}. Using default settings.")
        else:
            print(f"Config file {self.config_path} not found. Using default settings and creating it.")
            self._save_settings(loaded_settings) # Create with defaults if not exists
        return loaded_settings

    def _save_settings(self, settings_to_save):
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, indent=4, ensure_ascii=False)
            print(f"Settings saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving settings to {self.config_path}: {e}")

    def get_setting(self, key):
        return self.settings.get(key)

    def get_all_settings(self):
        return self.settings.copy()

    def update_setting(self, key, value):
        self.settings[key] = value
        self._save_settings(self.settings)

    def update_multiple_settings(self, new_settings_dict):
        self.settings.update(new_settings_dict)
        self._save_settings(self.settings)

    # Specific getters for convenience
    def get_openai_api_key(self):
        return self.settings.get("openai_api_key", "")

    def get_openai_model_name(self):
        return self.settings.get("openai_model_name", DEFAULT_SETTINGS["openai_model_name"])

    def get_openai_base_url(self):
        return self.settings.get("openai_base_url", "")

    def get_transcription_model_name(self):
        return self.settings.get("transcription_model_name", DEFAULT_SETTINGS["transcription_model_name"])
    
    def get_sherpa_onnx_model_dir(self):
        return self.settings.get("sherpa_onnx_model_dir", DEFAULT_SETTINGS["sherpa_onnx_model_dir"])

if __name__ == '__main__':
    sm = SettingsManager() # Manages settings in config.json in the current directory

    print("Current settings:")
    print(json.dumps(sm.get_all_settings(), indent=4))

    # Example: Update a setting
    # current_model = sm.get_openai_model_name()
    # new_model = "gpt-4" if current_model == "gpt-3.5-turbo" else "gpt-3.5-turbo"
    # print(f"\nUpdating OpenAI model name to: {new_model}")
    # sm.update_setting("openai_model_name", new_model)
    
    # print("\nSettings after update:")
    # print(json.dumps(sm.get_all_settings(), indent=4))

    # Example: Get a specific setting
    print(f"\nRetrieved OpenAI API Key: '{sm.get_openai_api_key()}' (should be empty by default)")
    print(f"Retrieved Transcription Model: '{sm.get_transcription_model_name()}'")

    # Example: Update multiple settings
    # print("\nUpdating multiple settings...")
    # sm.update_multiple_settings({
    #     "openai_api_key": "test_key_123",
    #     "openai_base_url": "https://custom.api.com/v1"
    # })
    # print("\nSettings after multiple updates:")
    # print(json.dumps(sm.get_all_settings(), indent=4))

    # Reset to defaults for next run if testing changes
    # print("\nResetting to default-like settings for next test run...")
    # sm.update_multiple_settings({
    #     "openai_api_key": "",
    #     "openai_base_url": "",
    #     "openai_model_name": DEFAULT_SETTINGS["openai_model_name"]
    # })
    # print("Final settings:")
    # print(json.dumps(sm.get_all_settings(), indent=4))