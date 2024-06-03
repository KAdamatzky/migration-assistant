from sqlglot import optimizer, expressions,transpile,parse_one

################################################################################
################################################################################

# stole this from the internet.
# this qualifies tables columns to their full names
def qualify_columns(expression):
    try:      
        expression = optimizer.qualify_tables.qualify_tables(expression)
        expression = optimizer.isolate_table_selects.isolate_table_selects(expression)
        expression = optimizer.qualify_columns.qualify_columns(expression)

    except:
        pass  
 
    return expression
################################################################################
################################################################################
  
# stole this from the internet.
# this returns a list of dictionaries of all tables and columns in those tables used in the provided sql. 
# dialect is set to tsql - could look at parameterising this. 
def parse_sql(sql_query, dialect='tsql'):

    ast = parse_one(sql_query, read=dialect)
    ast = qualify_columns(ast)
    
    physical_columns = {}
    
    for scope in optimizer.scope.traverse_scope(ast):
        for c in scope.columns:
            if isinstance(scope.sources.get(c.table), expressions.Table):
                table_info = scope.sources.get(c.table)
                full_table_name=table_info.catalog +'.'+table_info.db+'.'+table_info.name
                col_name = {c.name}
                try:
                    physical_columns[full_table_name].add(c.name)
                        
                except KeyError:
                    update = {full_table_name: col_name}
                    physical_columns.update(update)
                
    # convert output into expected input for next step
    return [{'table_name':k, 'columns': list(v)} for k,v in physical_columns.items()]


################################################################################
################################################################################
# this is called to do the T-SQL to DB-SQL translation. 
# this will return a list for each statement that goes into it - join all the statements together 
# before returning them as the gradio code component requires a string input
def sqlglot_transpilation(sql_query):
   transpiled = transpile(sql_query, read="tsql", write="spark", pretty=True)
   return "\n;\n".join(transpiled)



################################################################################
################################################################################

# Function for checking validity of SQL query. If there is an error, it returns  
# the description, which line the error is on, and the highlighted section.
def validity_check(input_query):
    try:
        #sqlglot.transpile(input_query, read="spark", write="spark")
        sqlglot.parse(input_query, read="spark")
        return("Valid")
    except sqlglot.errors.ParseError as e:
        errors = e.errors[0]
        #print(f"{errors['description']}\nLine: {errors['line']}\n{errors['start_context']}{errors['highlight']}{errors['end_context']}")
        #print(f"{errors['description']}\nLine: {errors['line']}\n{errors['start_context']}\n------------\n{errors['highlight']}\n^^^^^^^^^^^^{errors['end_context']}")
        #return(f"{errors['description']}\nLine: {errors['line']}")
        return(errors)
    
################################################################################
################################################################################

# takes in a dict of {table_name: str, columns: list} where the columns are the columns used from that table
# uses describe table to retrieve table and column descriptions from unity catalogue. 
# wrapped in a try statement in case table not found in UC to make it work on code only
def get_table_metadata(input_dict):

    try:
        table_name = input_dict["table_name"]
        columns = input_dict["columns"]
        details = execute_sql(cursor, f"describe table extended {table_name}")
        table_comment = f"This is the information about the {table_name} table. " + list(filter(lambda x: x.col_name == "Comment", details)).pop().data_type
        row_details = execute_sql(cursor, f"describe table {table_name}")
        row_details = ["The column " + x.col_name + " has the comment \"" + x.comment + "\"" for x in details if x.col_name in columns]
        row_details = " ".join(row_details)
        return table_comment + " " + row_details
    except:
        return ''




# use this to build to the initial prompt to get the intent
def build_table_metadata(sql_query):
    # get tables and columns
    table_info = parse_sql(sql_query, 'spark')
    # get table and column metadata
    table_metadata = []
    for x in table_info:
        table_details = get_table_metadata(x)
        if table_details != '':
            table_metadata.append(table_details)
    if table_metadata != []:
        # join up the metadata into a single string to add into the prompt
        table_column_descriptions = "\n\n ".join(table_metadata)
        return table_column_descriptions
    else:
        return None


