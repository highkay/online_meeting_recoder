# llm_summarizer.py
import openai
import os

class LLMSummarizer:
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo", base_url=None):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.client = None # ADDED

        if self.api_key: # MODIFIED BLOCK - Only init client if api_key is truthy
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url # base_url can be None here
            )
    
    def update_config(self, api_key_new=None, model_name_new=None, base_url_new=None): # ENTIRELY REPLACED METHOD
        client_needs_reinit = False
        if api_key_new: # Only update if api_key_new is truthy
            if self.api_key != api_key_new:
                self.api_key = api_key_new
                client_needs_reinit = True
        
        if model_name_new: # Only update if model_name_new is truthy
            if self.model_name != model_name_new:
                self.model_name = model_name_new
                # Model name change does not require client re-initialization by itself

        if base_url_new: # Only update if base_url_new is truthy
            if self.base_url != base_url_new:
                self.base_url = base_url_new
                client_needs_reinit = True
        
        # Re-initialize client if key/base_url changed, or if api_key is set but client is not yet initialized
        if client_needs_reinit or (self.api_key and not self.client):
            if self.api_key: # Ensure api_key is truthy before creating client
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            else: # If api_key became non-truthy (e.g. empty string after an update)
                self.client = None

    def summarize(self, transcript_text): # MODIFIED SIGNIFICANTLY
        """
        Generates a Markdown-formatted meeting summary from the transcript text.

        Args:
            transcript_text (str): The raw text transcript of the meeting.

        Returns:
            str: Markdown formatted meeting summary, or None if summarization fails.
        """
        # Ensure client is initialized, attempting to use env vars if not set
        if not self.client and not self.api_key: 
            env_api_key = os.getenv("OPENAI_API_KEY")
            if env_api_key:
                args_for_update = {'api_key_new': env_api_key}
                env_base_url = os.getenv("OPENAI_BASE_URL") # Check for base_url from env
                if env_base_url:
                    args_for_update['base_url_new'] = env_base_url
                self.update_config(**args_for_update) # This sets attributes and initializes client if api_key is truthy
            else:
                print("Error: OpenAI API key not configured and not found in OPENAI_API_KEY environment variable.")
                return "Error: API Key not provided. Please configure it in settings or via OPENAI_API_KEY."
        elif not self.client and self.api_key: # API key is set (truthy), but client is somehow not initialized
            # This could happen if __init__ had a truthy api_key but client init failed, or subsequent update_config made api_key non-truthy then truthy again.
            # Or if api_key was empty string initially.
            # Ensure client is created if api_key is now truthy.
            if self.api_key: # Double check api_key is truthy
                 self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            else: # api_key is set but not truthy (e.g. empty string)
                print("Error: OpenAI API key is non-truthy. Client not initialized.")
                return "Error: API Key is invalid (e.g. empty string). Client not initialized."

        if not self.client: # Final check, if client is still None, something is wrong with API key or config
            print("Error: OpenAI client could not be initialized. Check API key and configuration.")
            return "Error: OpenAI client not initialized. Please check API key and configuration."
        
        if not transcript_text or transcript_text.isspace():
            print("Error: Transcript text is empty.")
            return "Error: Transcript text is empty. Cannot generate summary."

        prompt = f"""
        请将以下会议转录文本转换为一份详细的会议纪要，使用Markdown格式。会议纪要应包含以下部分：

        1.  **会议主题**：根据内容判断，如果没有明确主题，则写“未定主题”。
        2.  **日期与时间**：如果文本中提及，请记录。否则留空或写“未提及”。
        3.  **参会人员**：如果文本中提及，请列出。否则留空或写“未提及”。
        4.  **主要议题与讨论**：详细记录会议讨论的各个主要议题和相关内容。
        5.  **关键决策与结论**：清晰列出会议中做出的重要决策和达成的结论。
        6.  **行动项（待办事项）**：明确指出分配给谁的什么任务，以及可能的截止日期。
        7.  **其他备注**：任何其他值得记录的要点。

        请确保格式清晰、专业，并准确反映会议内容。

        会议转录文本如下：
        ---BEGIN TRANSCRIPT---
        {transcript_text}
        ---END TRANSCRIPT---
        """

        try:
            print(f"Sending request to LLM. Model: {self.model_name}, Base URL: {self.base_url or 'Default OpenAI'}")
            response = self.client.chat.completions.create( # MODIFIED API call
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的会议纪要助手，擅长将会议记录整理成结构化的Markdown文档。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5, 
            )
            summary = response.choices[0].message.content.strip() # MODIFIED attribute access
            print("Summary generated successfully.")
            return summary
        except openai.AuthenticationError as e:
            print(f"OpenAI API Authentication Error: {e}")
            return f"Error: OpenAI API Authentication Failed. Please check your API key. Details: {e}"
        except openai.OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return f"Error: OpenAI API request failed. Details: {e}"
        except Exception as e:
            print(f"An unexpected error occurred during summarization: {e}")
            return f"Error: An unexpected error occurred. Details: {e}"

if __name__ == '__main__':
    # Example Usage:
    # Ensure you have OPENAI_API_KEY environment variable set or pass it to the constructor.
    # summarizer = LLMSummarizer(api_key="YOUR_API_KEY", model_name="gpt-3.5-turbo")
    summarizer = LLMSummarizer() # Tries to use env var or expects it to be set via update_config

    dummy_transcript = """
    张三：好了，我们开始今天的会议。主要是讨论一下下个季度产品A的推广计划。
    李四：我建议我们可以尝试在社交媒体上多做一些投入，特别是针对年轻用户群体。
    王五：同意李四的看法。另外，我觉得我们可以和一些KOL合作，提升品牌影响力。
    张三：好主意。那李四你负责调研一下社交媒体平台的数据，王五你来联系KOL资源。下周五之前给我一个初步方案。
    李四：没问题。
    王五：收到。
    赵六：关于预算方面，我们这个季度有多少额度？
    张三：预算大概是50万。具体分配我们下次再细化。
    张三：好，如果没有其他问题，今天就先到这里。散会。
    """

    print("Attempting to summarize dummy transcript...")
    summary_md = summarizer.summarize(dummy_transcript)

    if summary_md and not summary_md.startswith("Error:"):
        print("\n--- Meeting Summary (Markdown) ---")
        print(summary_md)
    else:
        print(f"\nFailed to generate summary: {summary_md}")
        print("Please ensure your OpenAI API key is set correctly (e.g., as an environment variable OPENAI_API_KEY) and the model/base_url are valid.")

    # Test with empty transcript
    print("\nAttempting to summarize empty transcript...")
    empty_summary = summarizer.summarize(" ")
    print(f"Summary for empty transcript: {empty_summary}")

    # Test with no API key (if not set in env)
    # temp_key = os.environ.pop("OPENAI_API_KEY", None)
    # summarizer_no_key = LLMSummarizer() # API key will be None
    # print("\nAttempting to summarize with no API key (should fail gracefully)...")
    # no_key_summary = summarizer_no_key.summarize(dummy_transcript)
    # print(f"Summary with no API key: {no_key_summary}")
    # if temp_key:
    #     os.environ["OPENAI_API_KEY"] = temp_key # Restore if it was there