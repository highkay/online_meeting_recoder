# llm_summarizer.py
import openai
import os

class LLMSummarizer:
    def __init__(self, api_key=None, model_name="gpt-3.5-turbo", base_url=None):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        
        if self.api_key:
            openai.api_key = self.api_key
        if self.base_url:
            openai.api_base = self.base_url

    def update_config(self, api_key=None, model_name=None, base_url=None):
        if api_key:
            self.api_key = api_key
            openai.api_key = self.api_key
        if model_name:
            self.model_name = model_name
        if base_url:
            self.base_url = base_url
            openai.api_base = self.base_url
        # If only base_url is updated, api_key might need to be re-set if it was None before
        elif self.api_key and not openai.api_key: 
             openai.api_key = self.api_key

    def summarize(self, transcript_text):
        """
        Generates a Markdown-formatted meeting summary from the transcript text.

        Args:
            transcript_text (str): The raw text transcript of the meeting.

        Returns:
            str: Markdown formatted meeting summary, or None if summarization fails.
        """
        if not self.api_key:
            print("Error: OpenAI API key not configured.")
            # Try to get from environment variable as a fallback
            env_api_key = os.getenv("OPENAI_API_KEY")
            if env_api_key:
                print("Found API key in OPENAI_API_KEY environment variable.")
                self.api_key = env_api_key
                openai.api_key = self.api_key
            else:
                return "Error: API Key not provided. Please configure it in settings."
        
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
            print(f"Sending request to LLM. Model: {self.model_name}, Base URL: {openai.api_base or 'Default OpenAI'}")
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "你是一个专业的会议纪要助手，擅长将会议记录整理成结构化的Markdown文档。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5, # Adjust for creativity vs. factuality
            )
            summary = response.choices[0].message['content'].strip()
            print("Summary generated successfully.")
            return summary
        except openai.error.AuthenticationError as e:
            print(f"OpenAI API Authentication Error: {e}")
            return f"Error: OpenAI API Authentication Failed. Please check your API key. Details: {e}"
        except openai.error.OpenAIError as e:
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