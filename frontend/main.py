# frontend/main.py
from typing import List
from django_unicorn.components import UnicornView

from bm.settings import TEST_USER_ID
from qf_sim.sim_runner import SimCore


class ChatView(UnicornView):
    template_name = r"C:\Users\wired\OneDrive\Desktop\BestBrain\frontend\templates\unicorn\i.html"

    message: str = ""
    chat_log: List[str] = []
    dots: List[dict] = []
    nodes:list = []

    def send(self):
        print("HI!")
        if not self.message:
            return
        self.chat_log.append(f"You: {self.message}")
        self.chat_log.append(f"Bot: Echo: {self.message}")
        self.message = ""


    def run_sim(self):
        test = SimCore(
            user_id=TEST_USER_ID,
            env_id=f"env_bare_{TEST_USER_ID}",
            visualize=False,
            demo=True
        )
        self.nodes = [attrs for nid, attrs in test.g.G.nodes(data=True)]
        test.run_connet_test()


