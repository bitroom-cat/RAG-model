import tiktoken
'''
def run():
    enc = tiktoken.get_encoding('cl100k_base')
    raw_text = "booking a room at baba balak nath."

    token_id = enc.encode(raw_text)

    print(token_id)
    print(len(token_id))

    for token in token_id:
        chunk = enc.decode_single_token_bytes(token).decode('utf-8')
        print(f"{token} ->'{chunk}'")



if __name__ == "__main__":
    run()   
'''
#phase 2 truncate prevent mass token usages by LLM

"""
def truncate(text: str, max_tokens: int) -> str:
    enc = tiktoken.get_encoding('cl100k_base')

   

    token_id = enc.encode(text)

   

    if len(token_id) > max_tokens:
        token_id = token_id[:max_tokens]
        token_id = token_id


    truncate_text = enc.decode(token_id)

    return truncate_text


test_string = "Booking a room at Baba Balak Nath."
safe_string = truncate(test_string, 4)

print(f"Original: '{test_string}'")
print(f"Truncated: '{safe_string}'")
   

#vector mathematics for text 

import numpy as np


def vector ():
    hotel = np.array([0.8,0.7,0.6])
    resort = np.array([0.9,0.9,0.8])
    motel = np.array([0.2,0.3,0.2])

    print(f"Hotel vector:{hotel}")
    print(f"resort vector:{resort}")

    diff_1 = np.abs(hotel - resort)
    diff2 = np.abs(hotel - motel)
    print("\nComparing semantic differences:")
    print(f"Hotel vs Resort difference: {np.sum(diff_1):.2f}")
    print(f"Hotel vs Motel difference: {np.sum(diff2):.2f}")

if __name__== "__main__":
    vector()    

def vector2 ():
    king = np.array([0.9,0.1,0.9])
    man = np.array([0.0,0.9,0.1]) 
    woman = np.array([0.0,0.1,0.9])


    dif = np.abs(king - man + woman) # king - man + women = queen

    print(f"Diff{np.sum(dif):.2f}")

#King ($0.9, 0.1, 0.9$): Might represent (Royalty, Masculinity, Power).
# Minus Man: You are "subtracting" the masculine features from the King.
# Plus Woman: You are "adding" feminine features.

if __name__ == "__main__":
    vector2()


import numpy as np 
from numpy.linalg import norm

def cosin():

    query = np.array([0.8,0.1,0.9,0.2])
    doc1 = np.array([0.9,0.2,0.8,0.1])
    doc2 = np.array([0.1,0.9,0.1,0.8])
    
    def calc(v1,v2):
        return np.dot(v1,v2) / ((norm(v1)) * (norm(v2)))
    
    score1 = calc(query,doc1)
    score2 = calc(query,doc2)

    print(f"doc1{score1:.4f}")
    print(f"doc1{score2:.4f}")


if __name__ == "__main__":
    cosin()    


import numpy as np
from numpy.linalg import norm


def calc(v1,v2):
    return np.dot(v1,v2) / ((norm(v1)) * (norm(v2)))

def search_database(query_vector, database_vectors):
    best =  None
    highest = -1.0

    for name, vector in database_vectors.items():
        score = calc(query_vector, vector)
        print(F"{name}:{score}")

        if score > highest:
            highest = score
            best_match = name



query_vector = np.array([0.5, 0.5, 0.5]) # User searches: "Family friendly accommodation"

database_vectors = {
    "hostel_record": np.array([0.1, 0.9, 0.1]),
    "family_suite_record": np.array([0.6, 0.5, 0.4]),
    "flight_record": np.array([0.9, 0.1, 0.1])
}

search_database(query_vector, database_vectors)



#The Data Pipeline (Ingestion & Embedding)

import os

def load(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"file not found{file_path}")
        return ""
    
    try:
        with open(file_path, 'r' , encoding = 'utf-8') as file:
            content = file.read()
            print(f"[Success] Loaded {len(content)} characters from {file_path}")
            return content
    except Exception as e :
            print(f"[Error] Failed to read {file_path}. Reason: {e}")
            return ""

if __name__ == "__main__":
    text = load("sample.txt")    



# batch load of documents

import os

def load(file_path : list)-> dict:
    document = {}
    for path in file_path:
        try:
            with open(path, 'r', encoding ='utf-8') as file:
                document[path] = file.read()
        except FileNotFoundError:
            document[path] = ""
    return document 
    



if __name__ == "__main__":
    vault = load(["room101.txt","room102.txt","missing.txt"])    

    print(vault)


#chunking of words


import os

def word(text:str, chunk_size:int, overlap: int) -> list:

    words = text.split()

    chunk = []
    i =0
    while i < len(words):
        slice = words[i:i+chunk_size]
        chunk_string = " ".join(slice)
        chunk.append(chunk_string)

        i += (chunk_size-overlap)

        if i + overlap >= len(words):
            break
        
    return chunk

if __name__ == "__main__":
    test = "BABA BAlak Nath temple is located in the hills"

    result = word(test,4,2)
    for idx , chunk in enumerate(result):
        print(f"chunk{idx +1}:'{chunk}'")

 
# delivering to llm model

import ollama

def ollama_embedding_demo():
    # You can use whatever model you currently have pulled. 
    # (e.g., 'llama3', 'mistral', or ideally a dedicated embedding model like 'nomic-embed-text')
    model_name = 'llama3' # Change this to the name of a model you have in Ollama
    text_chunk = "Room 101 features a king-sized bed and Wi-Fi."
    
    print(f"Requesting embedding from local Ollama ({model_name})...")
    
    # Call the embeddings endpoint
    response = ollama.embeddings(model=model_name, prompt=text_chunk)
    
    # The vector is stored in the 'embedding' key
    vector = response['embedding']
    
    print(f"\nText: '{text_chunk}'")
    print(f"Vector Dimensions (Length): {len(vector)}")
    print(f"First 5 values: {vector[:5]}")

if __name__ == "__main__":
    # Make sure your Ollama app/server is running in the background before executing this!
    ollama_embedding_demo()




import ollama 

def embed (chunk_list: list) -> dict:

    vector = {}

    print(f"starting embedding for {len(chunk_list)}chunks...")

    for chunk in chunk_list:
        try:
            response = ollama.embeddings(model='nomic-embed-text', prompt =chunk)

            vector[chunk] = response['embedding']
            print(f"Done: {chunk[:15]}...")

        except Exception as e :
            print(f"error here  {e}")


    return vector  



my_chunks = [
    "BABA BAlak Nath temple", 
    "Nath temple is located", 
    "is located in the", 
    "in the hills"
]

vector = embed(my_chunks)


import chromadb

def setup():
    client = chromadb.PersistentClient(path = "./my_db")
    collection = client.get_or_create_collection(name="rules")# 2. Create a Collection (Think of this as: CREATE TABLE temple_rules)
    print(collection.name)

    chunks = [
        "The temple is open from 6 AM to 8 PM.",
        "Photography is strictly prohibited inside the main shrine.",
        "Visitors must remove their shoes before entering."
    ]
    embeddings = [
        [0.1, 0.2, 0.3], 
        [0.8, 0.1, 0.1], 
        [0.2, 0.9, 0.4]
     ] 
    
    id = ["rule1","rule2","rule3"]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=id
    )
    print(f"Successfully inserted {collection.count()} records into the database.")

if __name__ == "__main__":
    setup()



import chromadb
import ollama

def seed_vector_database():

    client = chromadb.PersistentClient(path="./my_db")
    collection = client.get_or_create_collection(name="hotel")
    print(collection.name)

    hotel_features = [
        "Room 101 is a Single Deluxe Suite. It features a king-sized bed.",
        "Room 102 is a Family Executive Room near the prayer hall.",
        "We offer a complimentary breakfast buffet from 7 AM to 10 AM.",
        "The hotel provides a free shuttle service to the Baba Balak Nath temple."
    ]

    documents_list =[]
    embeddings = []
    ids = []

    print("Processing and embedding data....")

    for index, feature_text in enumerate(hotel_features):
        response = ollama.embeddings(model='nomic-embed-text', prompt = feature_text)
        vector = response['embedding']

        row_id = f"row: {index}"

        documents_list.append(feature_text)
        embeddings.append(vector)
        ids.append(row_id)

        print(f"processed: {row_id}")
    
    collection.add(
        documents=documents_list,
        embeddings=embeddings,
        ids=ids
    )
    print("-" * 30)
    print(f"Total rows in database: {collection.count()}")
    print("Database seeding complete!")

if __name__ == "__main__":
    seed_vector_database()    

"""
"""
import chromadb
import ollama


def retrieve():

    client = chromadb.PersistentClient(path="./my_db")
    collection = client.get_collection(name="hotel")

    user= "can i get room?"
    print(f"User asked: '{user}'\n")
    print("Embedding query and searching database...")

    query_respose = ollama.embeddings(model='nomic-embed-text', prompt = user )
    query_vector = query_respose['embedding']

    result = collection.query(
        query_embeddings=[query_vector],
        n_results=1
    )

    retrieved = result['documents'][0][0]
    distances = result['distances'][0][0]

    print(f"Best Match: '{retrieved}'")
    print(f"Distance Score: {distances:.4f} (Lower means mathematically closer)")

if __name__ == "__main__":
    retrieve()   



import chromadb
import ollama


def search_hotel(query:str):

    client = chromadb.PersistentClient(path="./my_db")

    try:
        collection = client.get_collection(name="hotel")
        print(f"Searching database for: '{query}'...")
        print("-" * 30)

        response = ollama.embeddings(model='nomic-embed-text',prompt=query)
        vectors = response['embedding']


        results = collection.query(
            query_embeddings=[vectors],
            n_results=2
        )

        for i, doc in enumerate(results['documents'][0]):
            distance = results['distances'][0][i]
            print(f"Match #{i+1} (Distance: {distance:.4f}):")
            print(f" >> {doc}\n")

    except ValueError:
        print(f"error:lalalalalala") 






if __name__=="__main__":
    search_hotel("what is food situation?")

import ollama

def generate(question:str) -> str:
    print("thinking")

    response= ollama.chat(model='llama3',messages=[

        {
            'role': 'user',
            'content': question,
        },
    ])

    answer = response['message']['content']

    print("-" * 30)
    print(f"AI Response:\n{answer}")
    print("-" * 30)

    return answer

if __name__=="__main__":
    user_input= "tell me about joke on coding "
    generate(user_input)

"""

import chromadb
import ollama

def build(data: str, user: str) -> str:

    template="""
you are "temple management system" a qprofessional and polite assistent .
Answer the USER QUESTION strictly based on the provided CONTEXT.
if the context doesn't have the answer, say please contact the temple admistration for more contact.

CONTENT{context}

USER QUESTION {question}

AGENT :



    return template.format(context=data, question=user)


def get(question: str):
    client = chromadb.PersistentClient(path= "./my_db")

    collection = client.get_collection(name="hotel")

    resp = ollama.embeddings(model='nomic-embed-text', prompt= question)

    vector = resp['embedding']

    result = collection.query(query_embeddings=[vector],
                              n_results=2)
    return "\n".join(result['documents'][0])

def ask(input: str):
    context = get(input)

    full_prompt = build(context, input)

    print("assistent here")

    response = ollama.chat(model="llama3", messages=[{
        'role':'user',
        'content':full_prompt
        }])
    
    print("-" * 30)
    print(response['message']['content'])
    print("-" * 30)

if __name__ == "__main__":
    query = "why india is black?"
    ask(query)

"""
#==============================================
# =====   Phase 5: Agentic Behavior & Production
#==============================================
"""
def check(type:str) ->str:
    print("checking rooms")

    live_db ={
        "single":"3 rooms avliable",
        "suite":"Fully booked",
    }
    return live_db.get(type.lower(), "unknown room type")

tool= {
    'type':'function',
    'function':{
        'name':'check room avaitability',
        'description':'check the live database for the number of avilable rooms of a specific type ',
        'parameter':{
            'type':'object',
            'properties':{
                'room_type':{
                    'type':'string',
                    'description':'the type of room(e.g. single, suit)'
                }
            },

            'required':['room_type']
        }
    }
}



# making tools so that LLM can get data from Real time data and can process it
#ReAct (Reason + Act).


def get_weather(location:str)->str:

    print("fetching the latest weather updates")


    if location.lower() == "temple":
        return "sunny, 25C"
    else:
        return "Weather data not Available"
    
weather_tool = {
    'type': 'function',
    'function': {
        'name': 'get_weather',
        'description': 'Get the current weather conditions for a specific location to help tourists and farmers plan their visit.',
        'parameters': {
            'type': 'object',
            'properties': {
                'location': {
                    'type': 'string',
                    'description': 'The name of the location or landmark (e.g., temple, hamirpur)'
                }
            },
            'required': ['location']
        }
    }
} 


"""

import os
import google.generativeai as genai

def get_weather(location: str)->str:
    """
    Get the current weather condition
    """

    print(f"executing for location{location}")

    if location.lower() == "temple":
        return "sunny , 25Celcius"
    return "Weather data not available"

def run():

    api_key="AIzaSyAwV3xNIhJwAzEMkrT1GHrmLEta1mTofHE"
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name= 'gemini-2.5-flash',
        tools=[get_weather]

    ) 

    query = "What is the weather like at temple today"

    print(f"user {query}")
    print("Agent is thinking ....")

    chat = model.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(query)

    print(response.text)

if __name__=="__main__":
    run()    
















 