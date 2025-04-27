import asyncio
import json

import requests
from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer

from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

class ChatConsumer(AsyncWebsocketConsumer): #WebsocketConsumer

    async def connect(self):
        # start here when client bot window is "load"
        # todo if works get code from init an cleint chat here and make async
        self.close_task = asyncio.create_task(self.close_after_timeout(1800))  # 1800 seconds = 30 minutes

        await self.accept()

    def get_initial_data(self, user):
        # Your existing logic to fetch initial data
        user_bot = user.bots.get(name=data)

        # ... rest of your logic ...


    def receive(self, text_data):
        data = json.loads(text_data)
        input_type = data["input_type"]
        if not input_type:
            print("REQUEST FAILED CAUSE NO INPUT TYPE PROVIDED...")
            self.send(text_data=json.dumps({"message": "Invalid input...", "status_code": 41 }))

        if input_type == "init":
            init_url = "https://wired66.pythonanywhere.com/client/init/"
            try:
                response = requests.post(init_url)
                self.send(text_data=json.dumps(response))
            except  Exception as e:
                print("ERROR WHILE WEBSOCKET UINIT REQUEST OCCURRED...", e)
        elif input_type == "chat":
            init_url = "https://wired66.pythonanywhere.com/client/chat/"
            try:
                response = requests.post(init_url)
                self.send(text_data=json.dumps(response))
            except  Exception as e:
                print("ERROR WHILE WEBSOCKET UINIT REQUEST OCCURRED...", e)



    def disconnect(self, close_code):
        async_to_sync(self.channel_layer.group_discard)(
            self.group_name,
            self.channel_name
        )
