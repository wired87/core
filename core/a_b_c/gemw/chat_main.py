import os

from openai import OpenAI
import base64
import mimetypes

import dotenv
dotenv.load_dotenv()

class Chat:
    """
    Multimodal Chat wrapper for OpenAI GPT-4o.
    Accepts multiple files (any type) and a user prompt.
    Automatically encodes and classifies file inputs.
    """

    def __init__(self, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model



    def ask(self, file_list, user_prompt: str):
        """
        file_list: list of tuples (filename, bytes)
        user_prompt: textual user prompt
        Returns: model text output
        """
        print(f"--- Starting query to {self.model} with {len(file_list)} files ---")

        content_parts = []
        for fname, fbytes in file_list:
            content_parts.append(self._build_content_part(fbytes, fname))

        # append main user instruction
        content_parts.append({"type": "text", "text": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content_parts}],
            temperature=0.3,
        )

        result = response.choices[0].message.content
        print("--- Query finished. Response received ---")
        print("Response preview:", (result[:80] + "...") if result else "No response")

        return result


# Example usage:
if __name__ == "__main__":
    # Example files
    with open(r"/sm/fermion/ferm_base.py", "rb") as f:
        py_bytes = f.read()
    with open(r"/sm/fermion/ferm_base.py", "rb") as f:
        img_bytes = f.read()

    chat = Chat()
    reply = chat.ask(
        [("example.py", py_bytes), ("diagram.png", img_bytes)],
        "Analyze the code and the image context together.",
    )
    print("\nFinal model reply:\n", reply)
