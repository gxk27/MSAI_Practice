import csv
from datetime import datetime
#from the toolboxes, it is importing a tool to help with the ML code
from sklearn.feature_extraction.text import CountVectorizer#turns words into numbers
from sklearn.naive_bayes import MultinomialNB#the brain and learns from those numbers
#imports operating system so it can talk to the code and vise versa
import os
#import Groq
from groq import Groq

def save_to_csv(user_input, bot_reply, filename="lumina_history.csv"):
    file_exists = os.path.isfile(filename)
    
    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "user", "lumina"])
        
        if not file_exists:
            writer.writeheader()  # only writes column names on first creation
        
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user_input,
            "lumina": bot_reply
        })

def show_history(filename="lumina_history.csv"):
    if not os.path.isfile(filename):
        print("No history yet!")
        return
    
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        spacers()
        print(f"{'TIMESTAMP':<22} {'YOU':<30} {'LUMINA':<50}")
        spacers()
        for row in reader:
            print(f"{row['timestamp']:<22} {row['user']:<30} {row['lumina'][:47] + '...' if len(row['lumina']) > 50 else row['lumina']:<50}")
        spacers()

#creating spacers so output is more organized
def spacers():
    for _ in range(15):
        print("* ", end="")
    print("")

#client variable is grabbing my secret API Key, which AI models need in order to function! That is how AI models make money; every computation costs something. This is a free version.    
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# I imported os and Groq, so now we are pulling a language model. Additionally, I am assigning key values and creating variables within the function. 
def ask_groq(user_input, conversation_history):
    # add user message to history
    conversation_history.append({"role": "user", "content": user_input})
    
    reply = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": f"You are a friendly chatbot named Lumina."},
            *conversation_history  # ← spread entire history here!
        ]
    )
    
    bot_reply = reply.choices[0].message.content
    # add bot response to history too!
    conversation_history.append({"role": "assistant", "content": bot_reply})
    return bot_reply


#list of tuples with types of 'greeting', 'help', and 'goodbye'
sayings = [
    ("hello", "greeting"),
    ("yo", "greeting"),
    ("whats up", "greeting"),
    ("whats your name", "greeting"),
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

#creating labels with 0 being the phrase "X" and "Y" being the type of phrase "1"
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

#grabbing the ask_groq function and creating a variable for it
def response(conversation_history, user_input):
    spacers()
    groq_reply = ask_groq(user_input, conversation_history)
    print(groq_reply)
    spacers()
    save_to_csv(user_input, groq_reply) 

#Will keep the history of the conversations between you and Lumina!
conversationHistory = []

while True:
    bot = input("Hello I am Lumina! What can I help you with? ")

    if bot.lower() in ["quit", "exit"]:
        spacers()
        day = datetime.now().strftime("%A")  # grabs the full day name
        print(f"Goodbye! Have a good {day}!")
        break
    
    elif bot.lower() in ["history", "show history"]: 
        show_history()
        continue
    response(conversationHistory, bot)