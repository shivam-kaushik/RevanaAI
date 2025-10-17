
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openai



def ask_chatgpt(question):
    """Ask ChatGPT a question and print the response (OpenAI v1.x)"""
    try:
        api_key = "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"
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment. Please set it in your .env file.")
            return
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            max_tokens=200
        )
        print(f"Question: {question}")
        print(f"ChatGPT Response: {response.choices[0].message.content}")
    except Exception as e:
        print(f"❌ ChatGPT API call failed: {e}")


if __name__ == "__main__":
    user_question = input("Enter your question for ChatGPT: ")
    ask_chatgpt(user_question)