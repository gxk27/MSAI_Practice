
#from the toolboxes, it is importing a tool to help with the ML code
from sklearn.feature_extraction.text import CountVectorizer#turns words into numbers
from sklearn.naive_bayes import MultinomialNB#the brain and learns from those numbers
#imports operating system so it can talk to the code and vise versa
import os
#importing google ai because it is free (OPENAI is not and wont run unless you add funds)
import google.generativeai as genai

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")
reply = model.generate_content("say hello!")
print(reply.text)

#list of tuples with types of 'greeting', 'help', and 'goodbye'
sayings = [
    ("hello", "greeting"),
    ("yo", "greeting"),
    ("whats up", "greeting"),
    ("whats good", "greeting"),
    ("hey how are you", "greeting"),
    ("top of the morning", "greeting"),
    ("good afternoon", "greeting"),
    ("good evening chap", "greeting"),
    ("help me", "help"),
    ("can you assist me", "help"),
    ("i need assistance", "help"),
    ("are you able to help", "help"),
    ("can you help me out", "help"),
    ("assist me with this", "help"),
    ("im heading out", "goodbye"),
    ("i could use your brains for this", "help"),
    ("see you later", "goodbye"),
    ("adios", "goodbye"),
    ("im out of here", "goodbye"),
    ("bye", "goodbye"),
    ("smell ya later", "goodbye"),
    ("ill see you when i see you", "goodbye")
]

#greating labels with 0 being the phrase"x" and 1 being the type of phrase "1"
# separate the phrases and labels
X = [saying[0] for saying in sayings]  # all the phrases
Y = [saying[1] for saying in sayings]  # all the labels

#taking the tool from the toolbox, creating an instance and has not been used yet
vectorizer = CountVectorizer()
#fit → the vectorizer looks at ALL your phrases and learns the vocabulary 
# transform → converts those phrases into numbers
X_vectorized = vectorizer.fit_transform(X)
#taking the tool into the code and hasnt been used yet
ml_model = MultinomialNB()
#X_vectorized → the numbers representing phrases
# Y → the correct labels
ml_model.fit(X_vectorized, Y)

#creating a new function that reads the ML code and input it into the actual code
def classify(bot):
    fitting = vectorizer.transform([bot]) 
    storage = ml_model.predict(fitting)
    return storage

def response(intent):
    if intent[0] == 'greeting':
        print("Hey There!")

    elif intent[0] == 'goodbye':
        print("Have a good one!")

    elif intent[0] == 'help':
        print("I can help you with whatever you need!")

    else:
        print("I do not undertsand")

while True:

    #Chatbot asking you what you need
    bot = input("What can I help you with?")

    if bot.lower() in ["quit", "exit"]:
        print("Exiting code.")
        break
    #calling the classify function and setting the parameter to "bot", so it reads your input
    intent = classify(bot)
    response(intent)