# talk/consumers.py
import json
from talk.chatresponse import *
#from talk.train.train_controller import getResponse

from channels.generic.websocket import WebsocketConsumer


class TalkConsumer(WebsocketConsumer):
   # def connect(self):
   #     self.accept()

   # def disconnect(self, close_code):
   #     pass

    # def receive(self, text_data):
    #    text_data_json = json.loads(text_data)
    #    message = text_data_json["message"]

    #    self.send(text_data=json.dumps({"message": message}))

	def connect(self):
		self.accept()

	def disconnect(self, close_code):
		self.close()

	def receive(self, text_data):
		text_data_json = json.loads(text_data)
		expression = text_data_json['expression']
		try:
			#text_return = getResponse(expression)
			#result = eval(expression)
			result = chatbot_response(expression)
		except Exception as e:
			result = "Invalid Expression"
		self.send(text_data=json.dumps({
			'result': result
		}))
