import random
import json
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#f=> bring the data from intents.json=>json loader
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

#data file has tags,hidden_size, input_size , output_size,all_words, model_state
input_size  = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words   = data['all_words']
tags        = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "GEHU bot"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    #Since BOW return as numpy so we use this
    X = torch.from_numpy(X).to(device)

    output = model(X)
    # we want to get it from dimension 0
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "Not known to me..."


if __name__ == "__main__":
    print("Hi!, This is GEHU bot, how can I help you? (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You      : ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print("GEHU bot : ",resp)

