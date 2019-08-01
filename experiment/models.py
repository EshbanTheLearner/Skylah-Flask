import mongoengine as me
import datetime

class Chat(me.Document):
    message = me.ListField(me.StringField())
    response = me.ListField(me.StringField())
    message_time = me.ListField(me.StringField())
    response_time = me.ListField(me.StringField())

class Session(me.EmbeddedDocument):
    start_time = me.StringField()
    end_time = me.StringField()
    chat = me.ReferenceField(Chat)

class User(me.Document):
    name = me.StringField()
    email = me.StringField()
    password = me.StringField()
    session = me.EmbeddedDocumentListField(Session)

me.connect("test_connect")

def register():
    chat = Chat()
    chat.message = []
    chat.response = []
    chat.message_time = []
    chat.response_time = []
    chat.save()

    session = Session()
    session.start_time = "<S_T>"
    session.end_time = "<E_T>"
    session.chat = chat

    user = User()
    user.name = "Eshban"
    user.email = "eshban@test.com"
    user.password = "test12345"
    user.session = [session]

    print(user.to_json(indent=2))

    return user

#def update(user):

'''
chat = Chat()
chat.message = ["Hi! How are you?", "What are you doing?"]
chat.response = ["I'm fine! How are you doing?", "Just talking to you"]

session = Session()
session.start_time = datetime.datetime.now()
session.chat = chat

#session.save()

eshban = User()
eshban.name = "Eshban Suleman"
eshban.email = "eshban@test.com"
eshban.password = "test12345"
eshban.session = session
eshban.save()

taymoor = User()
taymoor.name = "Taymoor Akbar"
taymoor.email = "taymoor@test.com"
taymoor.password = "test12345"
taymoor.save()

print(eshban.to_json(indent=2))
print("==================================================")
print(taymoor.to_json(indent=2))


eshban = User("eshban", "eshban@test.com", "test12345", session)
eshban.save()
print(eshban.to_json(indent=2))
'''

eshban = register().save()
#eshban.save()
print(eshban.to_json(indent=2))
print(eshban.session[-1].chat.id)

chat_1 = Chat()
chat_1.message = ["hello", "iam fine and you?"]
chat_1.response = ["hi. how are you today?", "iam fine as well"]
chat_1.message_time = ["1", "2"]
chat_1.response_time = ["1", "2"]
chat_1.save()

sess = Session()
'''
sess.chat.message = chat_1.message
sess.chat.message_time = chat_1.message_time
sess.chat.response = chat_1.response
sess.chat.response_time = chat_1.response_time
'''
sess.start_time = "<S_T_1>"
sess.end_time = "<S_T_1>"
sess.chat = chat_1

eshban.session.append(sess)
eshban.save()
print(eshban.to_json(indent=2))
eshban.session[-1].chat.message.append("yo!")
eshban.save()
print(eshban.session[-1].chat.message)