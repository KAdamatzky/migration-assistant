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

################################################################################
# DATABRICKS INFO
################################################################################
# Personal access token necessary for authenticating API requests. Stored using a secret
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
#DATABRICKS_HOST = os.environ["DATABRICKS_HOST"] # Not currently used. 

################################################################################
# VECTOR EMBEDDING SET UP
################################################################################
# details on the vector store holding the similarity information
vsc = VectorSearchClient(  
    workspace_url = "https://" + os.environ["DATABRICKS_HOST"],
    personal_access_token = DATABRICKS_TOKEN
) 

VECTOR_SEARCH_ENDPOINT_NAME = os.environ["VECTOR_SEARCH_ENDPOINT_NAME"]
vs_index_fullname= os.environ["VS_INDEX_FULLNAME"]
intent_table = os.environ["INTENT_TABLE"]

################################################################################
# LLM SET UP
################################################################################

# the URL of the serving endpoint
MODEL_SERVING_ENDPOINT_URL = f"https://{os.environ['DATABRICKS_HOST']}/serving-endpoints"

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url=MODEL_SERVING_ENDPOINT_URL
)

################################################################################
# SQL SET UP
################################################################################

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
# CHAT USED DURING TRANSLATION
################################################################################

rules = """You are an expert in multiple SQL dialects. You must follow these rules:
    - You may only reply with SQL code with no other text. 
    - References to a schema within a catalog are in the format catalog.schema. For example: `catalog_name`.`schema_name` when 'catalog_name' is the catalog and 'schema_name' is the schema. The catalog and schema MUST be surrounded with SEPARATE pairs of backticks, e.g.: `catalog_name`.`schema_name` NOT `catalog_name.schema_name`.
    - You must keep all original catalog, schema, table, and field names.
    - Convert all dates to dd-MMM-yyyy format using the date_format() function. 
    - The date_format() function should not be surrounded by backticks.
    - Subqueries must end with a semicolon.
    - ONLY if the original query uses temporary tables (e.g. "INTO #temptable"), re-write these as either CREATE OR REPLACE TEMPORARY VIEW or CTEs. 
    - Custom field names should be surrounded by backticks.
    - Square brackets must also be replaced with backticks.
    - Only if the original query contains DECLARE and SET statements, re-write them according to the following format:
        DECLARE VARIABLE variable TYPE DEFAULT value; For example: DECLARE VARIABLE number INT DEFAULT 9;
        SET VAR variable = value; For example: SET VAR number = 9;
    - Ensure queries do not have # or @ symbols. 
    """.strip()

original_translation_system_prompt = rules + "\nTranslate the following Transact SQL query to Databricks Spark SQL:"

translation_system_prompt = original_translation_system_prompt

# Initialise chat history. Initially just the system message.
translation_chat = [
        {"role": "system", "content": translation_system_prompt}
        ]

translation_attempts = 1 # Used in the 'translation chain' to keep track of how many invalid versions of the query were re-translated. Mostly for debugging.

iteration_limit = 3 # Maximum number of translation attempts.

################################################################################
# CHAT USED DURING INTENT
################################################################################

original_intent_system_prompt = "Your job is to explain the intent of this SQL code."

intent_system_prompt = original_intent_system_prompt

intent_chat = [
    {"role": "system", "content": intent_system_prompt}
]

# Bool for whether llm_intent is generating intent for the first time or refining the original intent
refine_bool = False

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
        logger.warning(f"Translation attempts: {translation_attempts}") # For debugging to see if translation attempts are being made or LLM is timing out

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

def llm_intent(sql_query = None):
    
    global intent_chat, intent_system_prompt
    # build the query prompt by adding code and metadata descriptions
    system_prompt = intent_system_prompt

    if(refine_bool == False):
        intent_chat.append({"role": "user", "content": f"This is the SQL code: {sql_query}"})

    # call the LLM end point.
    chat_completion = client.chat.completions.create(
        messages=intent_chat,
        model=os.environ["SERVED_MODEL_NAME"],
        max_tokens=int(os.environ["MAX_TOKENS"])
    )

    return chat_completion.choices[0].message.content
    
################################################################################
# SAVING INTENT AND CODE
################################################################################
# this writes the code & intent into the lookup table
def save_intent(code, intent):
    code_hash = hash(code) # Change to combination of code + intent?    
    intent = intent.replace("\"", "\'") # Replace any double quotation marks with single quotations to avoid syntax errors when writing to table.
    logger.warning(f"Intent: {intent}")
    
    existing_id = execute_sql(cursor, f"SELECT id FROM {intent_table} WHERE id = {code_hash}")
    
    logger.warning(existing_id)

    if not existing_id:
        cursor.execute(f"INSERT INTO {intent_table} VALUES ({code_hash}, \"{code}\", \"{intent}\")")
        gr.Info("Code and intent saved to catalog")
    else:
        raise gr.Error("Identical code found in the table")

################################################################################
# FINDING SIMILAR CODE
################################################################################
# this does a look up on the vector store to find the most similar code based on the intent
def get_similar_code(intent):    
    #intent = intent[0][-1] # Extracts just the intent explanation, without the mention of original code
    
    #results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
    results = vsc.get_index("", vs_index_fullname).similarity_search(
    query_text=intent,
    columns=["code", "intent"],
    num_results=1)
    docs = results.get('result', {}).get('data_array', [])
    return(docs[0][0], docs[0][1])

################################################################################
# REFINING CODE BASED ON USER REQUESTS
################################################################################
# Function for refining LLM output based on user request.
def refine_code(refine_msg, input_code, translated_code):    
    global translation_chat, translation_attempts

    new_system_prompt = f"Please improve the translation according to the user request. Only respond with a SQL query. Any comments must be commented using -- \n The original input code was:{input_code}\n Your translated code is:{translated_code}"

    translation_chat = [
            {"role": "system", "content": new_system_prompt},
            {"role": "user",  "content": refine_msg}
            ]

    logger.warning(translation_chat)
    new_code = llm_translate()
    #new_code = translation_chain(prompt)
    return(new_code)
   
# Function for refining LLM-generated intent 
def refine_intent(refine_msg, intent):
    global intent_chat, refine_bool
    refine_bool = True
    intent_chat.extend([{"role": "assistant", "content": intent},
                        {"role": "user", "content": refine_msg}])
    new_intent = llm_intent()
    return(new_intent)


################################################################################
# FUNCTIONS FOR BUTTONS
################################################################################
# Resets intent chat back to just the system prompt
def reset_intent_chat():
    global intent_chat, intent_system_prompt, refine_bool
    intent_chat = [
    {"role": "system", "content": intent_system_prompt}
    ]
    refine_bool = False
   

# Allows user to upload .sql file. Does not currently work when hosted through dbtunnel      
def read_sql_file(path):
    logger.warning("Reading uploaded file...{path}")
    with open(path.name) as fd:
        sql_code = fd.read()
    return(sql_code)

################################################################################
# POP UPS (Separate functions so can be used sequentially and be optional)
################################################################################
def saving_popup():
    gr.Info("Saving code and intent...")

def intent_reset_popup():
    gr.Info("Intent chat reset")

################################################################################
# GRADIO UI
################################################################################
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
### An AI assistant for converting T-SQL (Microsoft SQL Server) code to Spark SQL and describing query intent

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
            with gr.Row():
                
                # Textbox for editing translation system prompt
                system_prompt_box = gr.Text(
                    label = "Add/remove instructions for the LLM here.",
                    value = translation_system_prompt,
                    lines = 10
                )

                # Slider to set maximum number of re-translation attempts
                max_iterations = gr.Slider(1, 10, value = 3, step = 1, label = "Maximum re-translation iterations", info = "Choose the number of translation iterations the LLM will perform if the original translation is syntactically invalid.")

            # Functions for updating and reseting translation system prompts
            def update_system_prompt(changed_prompt):
                global translation_system_prompt
                translation_system_prompt = changed_prompt
                gr.Info("Translation prompt updated.")
                return(translation_system_prompt)
            
            def reset_system_prompt():
                global translation_system_prompt
                translation_system_prompt = original_translation_system_prompt
                gr.Info("Translation system prompt reset.")
                return(translation_system_prompt)
            
            # Function for updating the maximum number of re-translation attempts
            def update_max_iterations(iterations):
                global iteration_limit
                iteration_limit = iterations
                gr.Info(f"Translation iteration limit set to {iteration_limit}")
                pass

            
            with gr.Row():
                update_prompt_btn = gr.Button("Update translation prompt")
                reset_prompt_btn = gr.Button("Reset translation prompt")
        
        # Button behaviours
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
                        ,language='sql-msSQL' 
                        ,value=""
                        )
                
                translate_button = gr.Button("Translate") 
          
            with gr.Column():
                # divider subheader
                gr.Markdown(""" ### Your Code Translated to Spark-SQL""")
                
                # output box of the T-SQL translated to Spark SQL
                translated = gr.Code(
                    label="Your code translated to Spark SQL"
                    ,language="sql-sparkSQL"
                    )
                
                # Input for additional refinement of code. This request is sent back to the translation chain along with the original code translation.
                refine = gr.Textbox(label = "Refine the translated code",
                                 info = "Enter any changes you would like the LLM to make to the translated query")
        
        def reset_translation_attempts():
            global translation_attempts

            translation_attempts = 1
        
        # Button and textbox behaviours
        #upload_btn.upload(read_sql_file, upload_btn, input_code) # Re-enable once uploading is figured out
        translate_button.click(fn=reset_translation_attempts)
        translate_button.click(fn=translation_chain, inputs=input_code, outputs=translated)
        refine.submit(fn = refine_code, inputs = [refine, input_code, translated], outputs = translated)
        
        
################################################################################
#### ADVANCED INTENT SETTINGS
################################################################################
    with gr.Accordion(label="Advanced Intent Settings", open=False):
        with gr.Column():
            # Sets the system prompt differently depending on if user selects the use of metadata.
            with gr.Row():
              
                intent_prompt = gr.Text(
                    label="System prompt for LLM to generate code intent.",
                    value=intent_system_prompt,
                    lines = 3
                    )
                
                def update_intent_prompt(changed_prompt):
                    global intent_system_prompt
                    intent_system_prompt = changed_prompt
                    gr.Info("Intent system prompt updated.")
                    return(intent_system_prompt)
            
                def reset_intent_prompt():
                    global intent_system_prompt
                    intent_system_prompt = original_intent_system_prompt
                    gr.Info("Intent system prompt reset.")
                    return(intent_system_prompt)
                
            with gr.Row():
                update_intent_prompt_btn = gr.Button("Update intent prompt")
                reset_intent_prompt_btn = gr.Button("Reset intent prompt")

        # Button behaviours
        update_intent_prompt_btn.click(fn = update_intent_prompt, inputs = intent_prompt, outputs = intent_prompt)
        reset_intent_prompt_btn.click(fn = intent_reset_popup, outputs = intent_prompt)
        reset_intent_prompt_btn.click(fn = reset_intent_prompt, outputs = intent_prompt)
            
################################################################################
#### AI GENERATED INTENT PANE
################################################################################
    # divider subheader

#***If the intent is incorrect, please edit***. Once you are happy that the description is correct, please click the button below to save the intent. This will help the Department by making it easier to identify duplication of what people are doing. 

    with gr.Accordion(label="Intent Pane", open=True):
        gr.Markdown(""" ## AI generated intent of what your code aims to do. 
                    
                    Intent is determined by an LLM which uses the **input** code. You may edit the LLM prompt for generating intent in the `Advanced Intent Settings` pane.
                    
                    ***If the intent is incorrect, please edit***. Once you are happy that the description is correct, please click the button below to save the intent. This will help the Department by making it easier to identify duplication of what people are doing.""")
        
        explain_button = gr.Button("Explain code intent using AI")

        # Textbox for LLM generated intent of the code. This is editable as well. 
        explained = gr.Textbox(label="AI generated intent of your code.")

        # Input for refining generated intent 
        msg = gr.Textbox(label="Refine intent", info = "Input any additional instructions for the intent description here and the LLM will re-generate the intent.")

        # Button for clearing generated intent and user message boxes
        clear = gr.ClearButton([explained, msg], value = "Reset chat")

        # Button for saving code intent
        submit = gr.Button("Save code and intent")

        # When user clicks the 'Explained' button, the intent chat is reset and llm_intent is fed the input code
        explain_button.click(reset_intent_chat)
        explain_button.click(
            fn=llm_intent,
            inputs=input_code,
            outputs=explained
            )
        
        # When a user presses enter in the 'Refine intent' box, the 'refine_intent' function is called and explained intent box is updated.
        msg.submit(
            fn = refine_intent,
            inputs = [explained, msg],
            outputs = explained
        )
        
        # Button behaviours
        clear.click(fn = reset_intent_chat)
        submit.click(saving_popup) 
        submit.click(save_intent, inputs=[input_code, explained]) # Maybe use translated code instead of input

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
        , inputs=explained
        , outputs=[similar_code, similar_intent])
    

################################################################################
#### RUNNING THE APP
################################################################################
# this is necessary to get the app to run 
if __name__ == "__main__":
    demo.queue().launch(
    server_name=os.getenv("GRADIO_SERVER_NAME"), 
    server_port=int(os.getenv("GRADIO_SERVER_PORT"))
    #server_port=int(8080)
  )

