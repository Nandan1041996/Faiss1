import psycopg2
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq



def sql_connection():
    '''
    Establish a PostgreSQL connection.
    '''
    connection_string = 'postgres://postgres:postgres@localhost:5432/postgres'
    connection = psycopg2.connect(connection_string)
    cursor = connection.cursor()
    return cursor, connection

# Step 1: Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# def get_most_similar_question(query, top_k=1):
#     '''
#     Retrieve the most similar question from the database using vector similarity.

#     Args:
#         query: The query question to compare.
#         top_k: Number of similar results to retrieve.

#     Returns:
#         The most similar question from the database.
#     '''
#     # Generate the query embedding
#     query_embedding = model.encode(query)

#     # Convert embedding to a PostgreSQL-readable format
#     lst_embedding = str(query_embedding.tolist())

#     # SQL query to find the most similar question
#     sql_query = f"""
#     SELECT question FROM public.que_ans_embed
#     ORDER BY embedding <-> '{lst_embedding}'
#     LIMIT {top_k};
#     """
#     cursor, connection = sql_connection()
#     cursor.execute(sql_query)
#     results = cursor.fetchall()
#     connection.close()

#     if results:
#         return results[0][0]  # Return the first similar question
#     return None


prompt_temp = ''' 
You are an assistant to help check whether the two questions provided by the user are similar or not based on their meaning.
Please return only 'similar' if the questions are similar, or 'not similar' if the questions are not similar. 
Do not provide any additional explanation.

Question1: {question1}
Question2: {question2}

Answer:
'''

# Step 3: Create the chain
def get_chain(llm, prompt_template):
    '''
    Creates an LLMChain to process the similarity check.

    Args:
        llm: Pretrained language model.
        prompt_template: The prompt template for question similarity.

    Returns:
        An instance of LLMChain.
    '''
    prompt = PromptTemplate(
        input_variables=["question1", "question2"],
        template=prompt_template
    )
    return LLMChain(llm=llm, prompt=prompt )


def get_similarity_result(chain, question1, question2):
    '''
    Compare two questions using the LLMChain.

    Args:
        chain: The LLMChain created by `get_chain`.
        question1: The original query question.
        question2: The most similar question from the database

    '''
    # Run the chain with the two questions
    result = chain.run({"question1": question1, "question2": question2})
    print('result::',result)

    if result == 'similar':
        return 0
    else:
        return 1
    


# Step 4: Main execution
if __name__ == "__main__":
    # Load the LLM (e.g., OpenAI GPT)
    llm = ChatGroq(model='llama-3.1-70b-versatile',api_key='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',temperature=0,max_retries=2)

    # llm = OpenAI(temperature=0)

    # Create the similarity chain
    similarity_chain = get_chain(llm, prompt_temp)

    # Define the query question
    query_question = "hdnckbsjbsdlkbn kvawds"

    # Retrieve the most similar question from the database
    similar_question = get_most_similar_question(query_question, top_k=1)
    print('similarity_que::',similar_question)
    if not similar_question:
        print("No similar questions found in the database.")
    else:
        print(f"Query Question: {query_question}")
        print(f"Most Similar Question: {similar_question}")

        
        # Get the similarity result
        result = get_similarity_result(similarity_chain, query_question, similar_question)
        print('result:',result)

