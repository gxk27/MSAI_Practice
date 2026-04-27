#importing Natural Language ToolKit
import nltk
#downloading 'punkt_tab' which is essentially a language file from the internet.
nltk.download('punkt_tab')
#grabbing one tool from the toolbox NLTK
from nltk.tokenize import word_tokenize
#from the toolboxes, it is importing a tool to help with the ML code
from sklearn.feature_extraction.text import CountVectorizer#turns words into numbers
from sklearn.naive_bayes import MultinomialNB#the brain and learns from those numbers

#list of tuples regarding a greeting, help, and goodbye
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
model = MultinomialNB()
#X_vectorized → the numbers representing phrases
# Y → the correct labels
model.fit(X_vectorized, Y)

#creating a new function that reads the ML code and input it into the actual code
def classify(bot):
    fitting = vectorizer.transform([bot]) 
    storage = model.predict(fitting)
    return storage
#creating a funciton that will tokenize what the user inputs "reading the input based off keywords"
def tokensizer(bot):
    tokens = word_tokenize(bot)
    return tokens

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

    if bot.lower() == "quit":
        print("Exiting code.")
        break
    #calling the classify function and setting the parameter to "bot", so it reads your input
    intent = classify(bot)
    response(intent)

