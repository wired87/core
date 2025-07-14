# unicorn/components/chat.py
from typing import List
from django_unicorn.components import UnicornView
import random

class ChatView(UnicornView):
    template_name = r"C:\Users\wired\OneDrive\Desktop\BestBrain\frontend\templates\unicorn\i.html"

    message: str = ""
    chat_log: List[str] = []
    dots: List[dict] = []

    def send(self):
        print("HI!")
        if not self.message:
            return
        self.chat_log.append(f"You: {self.message}")
        self.chat_log.append(f"Bot: Echo: {self.message}")
        self.message = ""

    def spawn_dot(self):
        self.dots.append({
            "x": random.randint(0, 90),
            "y": random.randint(0, 90),
            "size": random.randint(5, 20),
        })
