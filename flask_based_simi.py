

import psycopg2
from sentence_transformers import SentenceTransformer
# from langchain.prompts import PromptTemplate
# from langchain.chains.llm import LLMChain
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain_groq import ChatGroq
from deep_translator import GoogleTranslator
# from langchain.memory import ConversationSummaryBufferMemory
import numpy as np
from flask import Flask,request,render_template,jsonify

app = Flask(__name__)

# from langchain.embeddings import HuggingFaceEmbeddings
# Initialize the embedding model
# embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

def sql_connection():
    "Establish a PostgreSQL connection."
    
    connection_string = 'postgres://postgres:postgres@localhost:5432/postgres'
    connection = psycopg2.connect(connection_string)
    cursor = connection.cursor()
    return cursor, connection



def cosine_similarity(A, B):
    "find out similarity between two vector"

    dot_product = np.dot(A, B)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)
    
    if not magnitude_A or not magnitude_B:
        return 0  # Avoid division by zero
    
    return dot_product / (magnitude_A * magnitude_B)


def vector_search(vector1,vector2):
    "Find out similarity score."
    similarity = cosine_similarity(vector1, vector2)
    print('similarity_score::',similarity)
    return similarity

def get_most_similar_question(query_embedding, top_k=1):
    '''
    Retrieve the most similar question from the database using vector similarity.
    Args:
        query: The query question to compare.
        top_k: Number of similar results to retrieve.
    Returns:
        The most similar question from the database.
    '''

    # Convert embedding to a PostgreSQL-readable format
    lst_embedding = str(query_embedding.tolist())

    # SQL query to find the most similar question
    sql_query = f"""
    SELECT answer, embedding FROM public.que_ans_embed
    ORDER BY embedding <-> '{lst_embedding}' asc
    LIMIT {top_k};
    """
    cursor, connection = sql_connection()
    cursor.execute(sql_query)
    results = cursor.fetchall()
    connection.close()

    if results:
        answer =results[0][0]
        vector =  np.array(eval(results[0][1]))

        similarity_score = vector_search(query_embedding,vector)

        return similarity_score,results[0][0],answer # Return the first similar question
    
    return None

def get_answer(similar_vector):
    "when similarity score is > 0.89"

    # SQL query to find the most similar question
    sql_query = f"""
    SELECT answer FROM public.que_ans_embed
    ORDER BY embedding <-> '{similar_vector}'
    LIMIT 1;
    """
    cursor, connection = sql_connection()
    cursor.execute(sql_query)
    results = cursor.fetchall()
    connection.close()

    if results:
        answer = results[0][0]
    return answer


def get_similar_questions(query_embedding):
    "Get similar question when have similarity score >0.4"

    lst_embedding = str(query_embedding.tolist())
        
    # SQL query to find the most similar question
    sql_query = f"""
    SELECT question,embedding FROM public.que_ans_embed
    ORDER BY embedding <-> '{lst_embedding}'
    LIMIT {3};
    """
    cursor, connection = sql_connection()
    cursor.execute(sql_query)
    results = cursor.fetchall()
    connection.close()

    if results:
        similar_que_index_lst = [] 
        for i,j in enumerate(results):
            # Retrieve the most similar question from the database
            similar_vector = np.array(eval(j[1]))
            similarity_score = vector_search(query_embedding,similar_vector)

            if similarity_score>0.4:
                similar_que_index_lst.append(i)

        similar_question = ''
        for index in similar_que_index_lst:
            similar_question+=(results[index][0]+',\n')
    else:
        similar_question = ''
    return similar_question


@app.route("/", methods=["GET", "POST"])
def ask():
    print('in')

    query_text = request.form.get('query_text')
    selected_language = request.form.get('selected_language')
    print('query_text::',query_text)
    print('selected_lang:',selected_language)

    if query_text:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query_text)

        # Retrieve the most similar question from the database
        similarity_score,similar_vector,ans = get_most_similar_question(query_embedding, top_k=1)
        
        if similarity_score>=0.8:
            answer = ans
            
        elif similarity_score >=0.4 and similarity_score<0.8:
            similar_question = get_similar_questions(query_embedding)
            answer = f'\n Similar questions i found: {similar_question}'
        else:
            answer = 'Not Found.'

        if selected_language != 'en':
            # To translate
            if selected_language in ['gu','hi','ta']:
                answer = GoogleTranslator(source='en', target = selected_language).translate(answer)

        print('answer::',answer)
        return render_template("index.html", response=answer)
        # return render_template("index.html", response=answer)
    return render_template("index.html", response=None)



# Step 4: Main execution
if __name__ == "__main__":
    app.run(debug=False,port=5002)
        




    


