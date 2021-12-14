# importing all required libraries
import telebot
from telethon.sync import TelegramClient
from telethon.tl.types import InputPeerUser, InputPeerChannel
from telethon import TelegramClient, sync, events
 
  
# get your api_id, api_hash, token
# from telegram as described above
api_id = '18381175' 
api_hash = '1ec757acde0f3dd822b8615bd5532ece'
token = '5074612612:AAGIPobP0Grs5V1kMtJnPyna2UoNrFStbNk'
message = "Working..."
 
# your phone number
phone = '+13605501934'
  
# creating a telegram session and assigning
# it to a variable client
client = TelegramClient('session', api_id, api_hash)
  
# connecting and building the session
client.connect()
 
# in case of script ran first time it will
# ask either to input token or otp sent to
# number or sent or your telegram id
if not client.is_user_authorized():
  
    client.send_code_request(phone)
     
    # signing in the client
    client.sign_in(phone, int(input('Enter the code: ')))

receiver = InputPeerUser(api_id, api_hash)
client.send_message(receiver, message, parse_mode='html')
try:
    # receiver user_id and access_hash, use
    # my user_id and access_hash for reference
    
     pass
    # sending message using telegram client
    
except Exception as e:
     
    # there may be many error coming in while like peer
    # error, wrong access_hash, flood_error, etc
    print(e);
 
# disconnecting the telegram session
client.disconnect()