import pickle
import torch
#from scripts.chatbot.train import SPECIAL_TOKENS, build_input_from_segments
#from scripts.chatbot.utils import get_dataset_personalities, download_pretrained_model
from scripts.chatbot.interact import top_filtering, sample_sequence

def loader():
    f = open("models/chatbot/personality.pickle", 'rb')
    personality = pickle.load(f)
    f.close()

    f = open("models/chatbot/model.pickle", 'rb')
    model = pickle.load(f)
    f.close()

    f = open("models/chatbot/tokenizer.pickle", 'rb')
    tokenizer = pickle.load(f)
    f.close()

    f = open("models/chatbot/args.pickle", 'rb')
    args = pickle.load(f)
    f.close()

    return personality, model, tokenizer, args

personality, model, tokenizer, args = loader()

def chat_run(raw_text):

    #personality, model, tokenizer, args = loader()

    history = []
    inputs = []
    outputs = []

    inputs.append(raw_text)

    if raw_text.lower() == 'bye':
        r = "Take care. Bye!"
        #print(r)
        return r

    history.append(tokenizer.encode(raw_text))
    with torch.no_grad():
        out_ids = sample_sequence(personality, history, tokenizer, model, args)
    history.append(out_ids)
    history = history[-(2*args.max_history+1):]
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    outputs.append(out_text)

    return out_text
