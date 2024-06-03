import yaml
import gradio as gr
from openai import OpenAI
import os

import gradio as gr
from databricks import sql
from databricks.vector_search.client import VectorSearchClient
from utils.sqlglotfunctions import *
import logging # For printing translation attempts in console (debugging)

# Setting up logger
logging.basicConfig
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# # personal access token necessary for authenticating API requests. Stored using a secret
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
#DATABRICKS_HOST = os.environ["DATABRICKS_HOST"] # Not currently used. 

# details on the vector store holding the similarity information
vsc = VectorSearchClient(  
    workspace_url = "https://" + os.environ["DATABRICKS_HOST"],
    personal_access_token = DATABRICKS_TOKEN
) 

VECTOR_SEARCH_ENDPOINT_NAME = os.environ["VECTOR_SEARCH_ENDPOINT_NAME"]
vs_index_fullname= os.environ["VS_INDEX_FULLNAME"]
intent_table = os.environ["INTENT_TABLE"]

# details for connecting to the llm endpoint

# the URL of the serving endpoint
MODEL_SERVING_ENDPOINT_URL = f"https://{os.environ['DATABRICKS_HOST']}/serving-endpoints"

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=MODEL_SERVING_ENDPOINT_URL
)

# create a connection to the sql warehouse
connection = sql.connect(
server_hostname = f"https://{os.environ['DATABRICKS_HOST']}",
http_path       = os.environ["SQL_WAREHOUSE_HTTP_PATH"],
access_token    = DATABRICKS_TOKEN
)
cursor = connection.cursor()

# little helper function to make executing sql simpler.
def execute_sql(cursor, sql):
    cursor.execute(sql)
    return cursor.fetchall()
    
################################################################################
################################################################################

################################################################################
# CHAT USED DURING TRANSLATION
################################################################################

rules = """You are an expert in multiple SQL dialects. 
    You may only reply with SQL code with no other text. 
    References to a schema within a catalog are in the format catalog.schema. For example: `catalog_name`.`schema_name` when 'catalog_name' is the catalog and 'schema_name' is the schema. The catalog and schema MUST be surrounded with SEPARATE pairs of backticks, e.g.: `catalog_name`.`schema_name` NOT `catalog_name.schema_name`.
    You must keep all original catalog, schema, table, and field names.
    Convert all dates to dd-MMM-yyyy format using the date_format() function. 
    The date_format() function should not be surrounded by backticks.
    Subqueries must end with a semicolon.
    ONLY if the original query uses temporary tables (e.g. "INTO #temptable"), re-write these as either CREATE OR REPLACE TEMPORARY VIEW or CTEs. 
    Custom field names should be surrounded by backticks.
    Square brackets must also be replaced with backticks.
    Only if the original query contains DECLARE and SET statements, re-write them according to the following format:
        DECLARE VARIABLE variable TYPE DEFAULT value; For example: DECLARE VARIABLE number INT DEFAULT 9;
        SET VAR variable = value; For example: SET VAR number = 9;
    Ensure queries do not have # or @ symbols. 
    """

original_translation_system_prompt = rules + "\nPlease translate the following Transact SQL query to Databricks Spark SQL:"

translation_system_prompt = original_translation_system_prompt

# Initialise chat history. Initially just the system message.
translation_chat = [
        {"role": "system", "content": translation_system_prompt}
        ]

translation_attempts = 1 # Used in the 'translation chain' to keep track of how many invalid versions of the query were re-translated. Mostly for debugging.

iteration_limit = 3 # Maximum number of translation attempts.

use_metadata = False 

################################################################################
# LLM TRANSLATION OF SQL CODE
################################################################################
# this is called to actually send a request and receive response from the llm endpoint.
def llm_translate(): 

    global translation_chat

    # call the LLM end point.
    chat_completion = client.chat.completions.create(
        messages=translation_chat,
        model=os.environ["SERVED_MODEL_NAME"],
        max_tokens=int(os.environ["MAX_TOKENS"]),
        temperature = 0.5
    )
            
    llm_answer = chat_completion.choices[0].message.content

    # Extract the code from in between the triple backticks (```), since LLM often prints the code like this. 
    # Also removes the 'sql' prefix always added by the LLM.
    if '```' in llm_answer:
        sql_answer = llm_answer.split("```")[1].replace("sql", "", 1) 
        sql_answer = sql_answer.replace("\\*", "*")  # Replace any escape backslashes produced by the LLM (sometimes returns \*)
        llm_answer = sql_answer
        return llm_answer
    else:
        return llm_answer

################################################################################
# LLM VERIFICATION OF SQL CODE VALIDITY
################################################################################
# This function asks the LLM to check whether the translated code
# is in a valid Databricks Spark SQL format.
# It is limited to 1 token and can only reply 'Valid' or 'Invalid'.

def llm_validity(input_query):
    validation_system_prompt = """Check if this code has valid Databricks Spark SQL Syntax. 
    The code must not have any square brackets, #, or @ symbols.
    Reply ONLY with the words Valid or Invalid"""

    validation_chat = [
        {"role": "system", "content": validation_system_prompt},
        {"role": "user",  "content": input_query}
        ]
    
    # call the LLM end point.
    chat_completion = client.chat.completions.create(
        messages=validation_chat,
        model=os.environ["SERVED_MODEL_NAME"],
        max_tokens=1,
        temperature = 0.8
    )

    llm_answer = chat_completion.choices[0].message.content

    return(llm_answer)

################################################################################
# TRANSLATION CHAIN/LOOP
################################################################################
# This function first asks the LLM to translate the input SQL query. The translated query
# is then passed into the `llm_validity` function. 
# If the function returns 'Valid', the SQL query is returned to the user. 
# If the function returns 'Invalid' (or any other string), the translated query is passed back into
# the translation LLM and the loop continues until the validation LLM returns 'Valid'.

def translation_chain(sql_query):

    global translation_chat, translation_attempts

    # If this is the first input, chat history is refreshed and contains just the translation prompt and user query.
    if translation_attempts == 1:
        translation_chat = [
            {"role": "system", "content": translation_system_prompt},
            {"role": "user",  "content": sql_query}
            ]
        
    # Runs the translation LLM with the current chat history
    llm_answer = llm_translate()

    # Checks whether LLM answer is valid Spark SQL code
    #validity = validity_check(llm_answer) # This is the SQLGlot function (however, this sometimes lets through invalid code)
    validity = llm_validity(llm_answer)

    # While the code is not valid, adds the answer + error to chat list and asks to re-write code with error in mind.
    while((validity.lower() != "valid") & (translation_attempts < iteration_limit)): # Limit to 3 translation attempts by default

        fix_code_prompt = rules + "The following Databricks Spark SQL query contains an error. Please fix it without including any explanation or other text except for the query. Ensure there are none of the following symbols: # @ \\ / (hash, at sign, backward slash, forward slash)"
        
        translation_attempts += 1
        logger.warning(f"Translation attempts: {translation_attempts}")
        #gr.Info(f"Translation was inavlid. Re-trying. Attempt {translation_attempts}")

        # Resets chat history so it only contains new system prompt and LLM's original answer
        translation_chat = [
        {"role": "system", "content": fix_code_prompt},
        {"role": "user", "content": llm_answer}
        ]

        llm_answer = llm_translate()
        logger.warning(llm_answer)

        validity = llm_validity(llm_answer)
    # Once the code is valid, reset the chat history and return the translated code.
    else:
        gr.Info(f"Translated query after {translation_attempts} attempts")
        translation_attempts = 1
        translation_chat = [
        {"role": "system", "content": translation_system_prompt},
        {"role": "user", "content": sql_query}
        ]
        return(llm_answer)

################################################################################
# FUNCTIONS FOR  INTENT
################################################################################

def llm_intent(#metadata_prompt, 
               no_metadata_prompt, 
               sql_query):
    if(use_metadata):
        table_descriptions = build_table_metadata(sql_query)

        if table_descriptions:
            # set the system prompt
            system_prompt = metadata_prompt
            # build the query prompt by adding code and metadata descriptions
            query_prompt = f"This is the SQL code: {sql_query}. \n\n{table_descriptions}"
    else:
        system_prompt = no_metadata_prompt
        # build the query prompt by adding code and metadata descriptions
        query_prompt = f"This is the SQL code: {sql_query}"

    # call the LLM end point.
    chat_completion = client.chat.completions.create(
        messages=[
        {"role": "system", "content": system_prompt}
        ,{"role": "user",  "content": query_prompt}
        ],
        model=os.environ["SERVED_MODEL_NAME"],
        max_tokens=int(os.environ["MAX_TOKENS"])
    )

    # helpful for debugging -show the query sent to the LLM        
    #return [chat_completion.choices[0].message.content, query_prompt]
    # this is the return without the chat interface
    #return chat_completion.choices[0].message.content
    # this is the return for the chatbot - empty string to fill in the msg, list of lists for the chatbot
    return "", [[query_prompt, chat_completion.choices[0].message.content]]
# this is called to actually send a request and receive response from the llm endpoint.
def call_llm_for_chat(chat_history, 
                      query, 
                      #metadata_prompt, 
                      no_metadata_prompt, 
                      sql_query):
    if(use_metadata):
        table_descriptions = build_table_metadata(sql_query)

        if table_descriptions:
            # set the system prompt
            system_prompt = metadata_prompt
            # build the query prompt by adding code and metadata descriptions
            query_prompt = f"This is the SQL code:\n{sql_query}. \n\n{table_descriptions}"
    else:
        system_prompt = no_metadata_prompt
        # build the query prompt by adding code and metadata descriptions
        query_prompt = f"This is the SQL code: {sql_query}"

    #system_prompt = "You are a chatbot which helps users explain the intent of their code."
    messages=[
        {"role": "system", "content": system_prompt}
        ] 
    for q, a in chat_history:
      messages.extend(
        [{"role": "user",  "content": q}
        ,{"role": "assistant",  "content": a}]
        )
    messages.append(
        {"role": "user",  "content": query})
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=os.environ["SERVED_MODEL_NAME"],
        max_tokens=int(os.environ["MAX_TOKENS"])
    )
    return chat_completion.choices[0].message.content
    
################################################################################
################################################################################
# this writes the code & intent into the lookup table
def save_intent(code, intent):
    code_hash = hash(code) # Change to combination of code + intent?    
    intent = intent[0][-1] # Extracts just the intent explanation, without the mention of original code
    logger.warning(f"Intent: {intent}")

    existing_id = execute_sql(cursor, f"SELECT id FROM {intent_table} WHERE id = {code_hash}")

    logger.warning(existing_id)

    if not existing_id:
        cursor.execute(f"INSERT INTO {intent_table} VALUES ({code_hash}, \"{code}\", \'{intent}\')")
        gr.Info("Code and intent saved to catalog")
    else:
        raise gr.Error("Identical code found in the table")

################################################################################
################################################################################
# this does a look up on the vector store to find the most similar code based on the intent
def get_similar_code(intent):    
    intent = intent[0][-1] # Extracts just the intent explanation, without the mention of original code
    
    #results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
    results = vsc.get_index("", vs_index_fullname).similarity_search(
    query_text=intent,
    columns=["code", "intent"],
    num_results=1)
    docs = results.get('result', {}).get('data_array', [])
    return(docs[0][0], docs[0][1])

################################################################################
################################################################################
# Function for refining LLM output based on user request.
def refine_code(refine_msg, input_code, translated_code):    
    global translation_chat, translation_attempts
    new_system_prompt = f"Please improve the translation according to the user request.Only respond with a SQL query. Any comments must be commented using -- \n The original input code was:{input_code}\n Your translated code is:{translated_code}"
    #prompt = f"Please improve the translation according to the user request.\n The original input code was:{input_code}\n Your translated code is:{translated_code},\n User's refinement request: {refine_msg}"
    translation_chat = [
            {"role": "system", "content": new_system_prompt},
            {"role": "user",  "content": refine_msg}
            ]

    logger.warning(translation_chat)
    new_code = llm_translate()
    #new_code = translation_chain(prompt)
    return(new_code)
   

################################################################################
################################################################################

#Allows user to upload .sql file      
def read_sql_file(path):
    logger.warning("Reading uploaded file...{path}")
    with open(path.name) as fd:
        sql_code = fd.read()
    return(sql_code)
################################################################################
################################################################################
def saving_popup():
    gr.Info("Saving code and intent...")
################################################################################
################################################################################

# GRADIO UI

# this is the app UI. it uses gradio blocks https://www.gradio.app/docs/gradio/blocks
# each gr.{} call adds a new element to UI, top to bottom. 

# Potential themes:
    #theme=gr.themes.Soft()
    #theme = 'gradio/base'
    #theme = 'zenafey/zinc'
    #theme = 'upsatwal/mlsc_tiet'

with gr.Blocks(theme = 'zenafey/zinc') as demo:
    # title with Databricks image
    gr.Markdown("""<img align="right" src="https://asset.brandfetch.io/idSUrLOWbH/idm22kWNaH.png" alt="logo" width="120">

# SQL Migration Tool
### A context aware migration assistant for converting T-SQL (Microsoft SQL Server) code to Spark SQL and describing query intent

This application uses the DBRX-instruct LLM model to translate Transact SQL queries into Spark SQL format for use in the Databricks environment. Additionally, the LLM model can describe the intent of your query and find code with similar intent submitted by other analysts. 
\nIf you are happy with the code and intent generated, please press the `Save code and intent` button as this will enhance the `Find similar code` feature and help other users. 
\nThis tool is still in development, so any feedback is welcome.
\nPlease email Kristian.ADAMATZKY@EDUCATION.GOV.UK with any errors, feature suggestions, or other comments regarding this tool.  
""")
    
################################################################################
#### ADVANCED TRANSLATION OPTIONS PANE
################################################################################
    with gr.Accordion(label="Advanced Translation Settings", open=False):
        with gr.Column():
            # select SQL flavour
            """
            sql_flavour = gr.Dropdown(
                label = "Input SQL Type. Select SQL if unknown."
                ,choices = [
                     ("SQL", 'sql')
                    ,("Transact SQL", 'sql-msSQL')
                    ,("MYSQL"       , 'sql-mySQL')
                    ,("SQLITE"      , 'sql-sqlite')
                    ,("PL/SQL"      , 'sql-plSQL')
                    ,("HiveQL"      , 'sql-hive')
                    ,("PostgreSQL"  , 'sql-pgSQL')
                    ,("Spark SQL"   , 'sql-sparkSQL')
                    ]
                ,value="sql-msSQL"
            )
            
            # this function updates the code formatting box to use the selected sql flavour
            def update_input_code_box(language):
                input_code = gr.Code(
                    label="Input SQL"
                    ,language=language
                    )
                return input_code
            """
            with gr.Row():

                system_prompt_box = gr.Text(
                    label = "Add/remove instructions for the LLM here.",
                    value = translation_system_prompt,
                    lines = 10
                )

                max_iterations = gr.Slider(1, 10, value = 3, step = 1, label = "Maximum re-translation iterations", info = "Choose the number of translation iterations the LLM will perform if the original translation is syntactically invalid.")

            def update_system_prompt(changed_prompt):
                global translation_system_prompt
                translation_system_prompt = changed_prompt
                gr.Info("System prompt updated.")
                return(translation_system_prompt)
            
            def reset_system_prompt():
                global translation_system_prompt
                translation_system_prompt = original_translation_system_prompt
                gr.Info("System prompt reset.")
                return(translation_system_prompt)
            
            def update_max_iterations(iterations):
                global iteration_limit
                iteration_limit = iterations
                gr.Info(f"Translation iteration limit set to {iteration_limit}")
                pass

            

            with gr.Row():
                update_prompt_btn = gr.Button("Update system prompt")
                reset_prompt_btn = gr.Button("Reset system prompt")
            
            # Re-enable once uploading is figured out
            #upload_btn = gr.UploadButton("Upload a SQL file (currently does not work on Databricks)", file_types = ['.sql'], file_count="single") 

            #file_output = gr.File() # Used to test upload capabilities. Does not work when hosted using dbtunnel but works otherwise.

            #uploaded_sql = gr.Code("")
        
        update_prompt_btn.click(fn = update_system_prompt, inputs = system_prompt_box, outputs = system_prompt_box)
        reset_prompt_btn.click(fn = reset_system_prompt, outputs = system_prompt_box)
        max_iterations.input(update_max_iterations, max_iterations)
        
################################################################################
#### TRANSLATION PANE
################################################################################
    # subheader
    with gr.Accordion(label="Translation Pane", open=True):
        gr.Markdown("""Translation is performed by an LLM, therefore the output code may not always be correct. In which case, try translating your query again or modifying the system prompt in the `Advanced Translation Settings` pane above.
                    \nThe output code syntax is validated by an LLM before being returned so may undergo several iterations before you see a result. The maximum number of iterations can be set in the `Advanced Translation Settings` pane above.
                    \nTranslation may take a while so please be patient, especially with large queries.""")

        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    """ ### Input your T-SQL code here for translation to Spark-SQL."""
                    )
                
                # input box for SQL code with nice formatting
                input_code = gr.Code(
                        label="Input SQL"
                        ,language='sql-msSQL' # Does this need to be changed to `sql_flavour` if that option is re-enabled?
                        ,value=""
                        )
                
                translate_button = gr.Button("Translate") 

            # the input code box gets updated when a user changes a setting in the Advanced section
                #sql_flavour.input(update_input_code_box, sql_flavour, input_code)
            
            with gr.Column():
                # divider subheader
                gr.Markdown(""" ### Your Code Translated to Spark-SQL""")
                
                # output box of the T-SQL translated to Spark SQL
                translated = gr.Code(
                    label="Your code translated to Spark SQL"
                    ,language="sql-sparkSQL"
                    )
                
                # Input for additional refinement of code. This request is sent back to the translation chain along with the original code translation.
                refine = gr.Text(label = "Refine the translated code",
                                 value = "",
                                 lines = 2)
                
                refine_button = gr.Button("Refine code")
        
        def reset_translation_attempts():
            global translation_attempts

            translation_attempts = 1
        
        # Button behaviours
        #upload_btn.upload(read_sql_file, upload_btn, input_code) # Re-enable once uploading is figured out
        translate_button.click(fn=reset_translation_attempts)
        translate_button.click(fn=translation_chain, inputs=input_code, outputs=translated)
        refine_button.click(fn = refine_code, inputs = [refine, input_code, translated], outputs = translated)
        
        
################################################################################
#### ADVANCED INTENT SETTINGS
################################################################################
    with gr.Accordion(label="Advanced Intent Settings", open=False):
        with gr.Column():
            # select whether to use table metadata    
            '''
            use_table_metadata = gr.Checkbox(
                label = "Use table metadata if available (currently not working)",
                value = False,
                interactive = True
            )
            '''
            # Sets the system prompt differently depending on if user selects the use of metadata.
            with gr.Row():
                '''
                llm_sys_prompt_metadata = gr.Textbox(
                    label="System prompt for LLM to generate code intent if table metadata present."
                    ,value="""
                        Your job is to explain the intent of a SQL query. You are provided with the SQL Code and a summary of the information contained within the tables queried, and details about which columns are used from which table in the query. From the information about the tables and columns, you will infer what the query is intending to do.
                        """.strip(),
                    lines = 3
                    )
                '''
                llm_sys_prompt_no_metadata = gr.Textbox(
                    label="System prompt for LLM to generate code intent."
                    ,value="""
                        Your job is to explain the intent of this SQL code.
                        """.strip(),
                    lines = 3
                    )
                
                def switch_metadata_bool(bool):
                    global use_metadata
                    use_metadata = bool

                #use_table_metadata.change(fn = switch_metadata_bool, inputs = use_table_metadata)
        
################################################################################
#### AI GENERATED INTENT PANE
################################################################################
    # divider subheader

#***If the intent is incorrect, please edit***. Once you are happy that the description is correct, please click the button below to save the intent. This will help the Department by making it easier to identify duplication of what people are doing. 

    with gr.Accordion(label="Intent Pane", open=True):
        gr.Markdown(""" ## AI generated intent of what your code aims to do. 
                    
                    Intent is determined by an LLM which uses the **input** code. You may edit the LLM prompt for generating intent in the `Advanced Intent Settings` pane.""")
        # a box to give the LLM generated intent of the code. This is editable as well. 
        explain_button = gr.Button("Explain code intent using AI")
        explained = gr.Textbox(label="AI generated intent of your code.", visible=False)

        chatbot = gr.Chatbot(
            label = "AI Chatbot for Intent Extraction"
            ,height="70%"
            )
        
        msg = gr.Textbox(label="Chat", info = "Input any additional instructions for the intent description here and the LLM will re-generate the intent.")
        clear = gr.ClearButton([msg, chatbot], value = "Clear chat")
        submit = gr.Button("Save code and intent")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def respond(chat_history, 
                    message, 
                    #metadata_prompt, 
                    no_metadata_prompt, 
                    sql_query):                
            bot_message = call_llm_for_chat(chat_history, 
                                            message, 
                                            #metadata_prompt, 
                                            no_metadata_prompt, 
                                            sql_query)
            
            chat_history.append([message, bot_message])
            return "", chat_history
        
        explain_button.click(
            fn=llm_intent
            , inputs=[
                #llm_sys_prompt_metadata,
                llm_sys_prompt_no_metadata
                , input_code 
                ]
            , outputs=[msg, chatbot]
            
            )
        # msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        #     respond, chatbot, chatbot
        # )
        msg.submit(
            fn=respond
            ,inputs = [
                chatbot,
                msg,
                #llm_sys_prompt_metadata,
                llm_sys_prompt_no_metadata,
                input_code 
                ],
            outputs= [msg, chatbot]
        )
        
        submit.click(saving_popup) 
        submit.click(save_intent, inputs=[input_code, chatbot]) # Maybe use translated code instead of input

################################################################################
#### SIMILAR CODE PANE
################################################################################
    # divider subheader

    with gr.Accordion(label="Similar Code Pane", open=True):
        gr.Markdown(""" ## Similar code 
                    This code is thought to be similar to what you are doing, based on comparing the **intent** of your code with the intent of this code.
                    """)    
        
        find_similar_code=gr.Button("Find similar code")
        # a row with an code and text box to show the similar code
        with gr.Row():
            similar_code = gr.Code(
                label="Similar code to yours."
                ,language="sql-sparkSQL"
                )
            similar_intent = gr.Textbox(label="The similar codes intent.")

        

        
    find_similar_code.click(
        fn=get_similar_code
        , inputs=chatbot
        , outputs=[similar_code, similar_intent])
    
    
# this is necessary to get the app to run 
if __name__ == "__main__":
    demo.queue().launch(
    server_name=os.getenv("GRADIO_SERVER_NAME"), 
    server_port=int(os.getenv("GRADIO_SERVER_PORT"))
    #server_port=int(8080)
  )

