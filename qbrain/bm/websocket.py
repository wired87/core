import asyncio
import json

import requests
from channels.generic.websocket import WebsocketConsumer, AsyncWebsocketConsumer

from channels.generic.websocket import WebsocketConsumer
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer


