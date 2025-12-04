from collections import defaultdict
from neo4j import GraphDatabase
import time
import re
import all_connected_yangs as acy # Need this file if you are executing in the second way.
import os 



def create_node(driver, database, node_label, node_name, node_datatype, node_parent_name, node_parent_type, node_mod_belongs, node_value, node_description):
    """
    Create a new node in the Neo4j database.

    Parameters:
    driver: The Neo4j driver instance to connect to the database.
    database: The name of the database where the node will be created.
    node_label: The label for the node (e.g., module or type).
    node_name: The name property of the node.
    node_datatype: The datatype property of the node.
    node_parent_name: The name of the parent node.
    node_parent_type: The type of the parent node.
    node_mod_belongs: The module to which the node belongs.
    node_value: The value associated with the node.
    node_description: A description of the node.
    """
    # Define the query to create a node with the provided label and properties
    query = f"""
    MERGE (n:{node_label} {{ name: $name, datatype: $datatype, parent_name: $parent_name,
        parent_type: $parent_type, belongs: $belongs, value: $value, description: $description, label: $label }})
    """

    # Execute the query using the driver
    try: 
        with driver.session(database=database) as session:
            print(f"Executing query:\n{query}\n")
            session.run(query, 
                        name=node_name,           # The name property of the node
                        datatype=node_datatype,   # The datatype property of the node
                        parent_name=node_parent_name,  # Parent name
                        parent_type=node_parent_type,  # Parent type
                        belongs=node_mod_belongs,      # Module to which the node belongs
                        value=node_value,               # Value associated with the node
                        description=node_description,   # Description of the node
                        label=node_label)               # The label property to be added
            print(f"Node created - name: {node_name}, type: {node_label}, parent_name: {node_parent_name}, module_belongs: {node_mod_belongs}\n")
    except Exception as e:
        print("An error occurred while creating the node:", e)




def create_relationship(driver, database, parent_type, parent_name, child_type, child_name, relationship_type, belongs):
    """
    Create a relationship between two nodes in the Neo4j database, preventing duplicates.

    Parameters:
    driver: The Neo4j driver instance to connect to the database.
    database: The name of the database where the relationship will be created.
    parent_type: The label/type of the parent node (e.g., Person).
    parent_name: The name of the parent node (e.g., Alice).
    child_type: The label/type of the child node (e.g., Person).
    child_name: The name of the child node (e.g., Bob).
    relationship_type: The type of relationship to create (e.g., KNOWS).
    """
    # Define the query to match the two nodes and create a relationship if it doesn't exist
    if(child_type != 'None'):
        query = f"""
    MATCH (p1:{parent_type} {{name: $parent_name, belongs: $belongs}}), (p2:{child_type} {{name: $child_name, belongs: $belongs}})
    MERGE (p1)-[:{relationship_type}]->(p2)
    """
    else:
        query = f"""
    MATCH (p1:{parent_type} {{name: $parent_name, belongs: $belongs}}), (p2 {{name: $child_name, belongs: $belongs}})
    MERGE (p1)-[:{relationship_type}]->(p2)
    """

    # Execute the query in the provided database context
    try:
        with driver.session(database=database) as session:
            print(f"{query}\n")
            session.run(query, parent_name=parent_name, child_name=child_name, belongs = belongs)
            print(f"Relationship '{relationship_type}' created between {parent_name} and {child_name}. \n \n")
    except Exception as e:
        print("An error occurred while creating the relationship:", e)


    
def get_group_children(driver, database, node_label, group_name, mod_belongs):
    query = f"""
    MATCH (p:{node_label} {{name: $group_name, belongs: $mod_belongs}})-[]->(child)
    RETURN child
    """
    
    try:
        # Execute the query using the driver
        with driver.session(database=database) as session:
            result = session.run(query, group_name=group_name, mod_belongs = mod_belongs)
            # children = [record['properties']['name'] for record in result]
            children = []
            for record in result:
                child_node = record['child']  # This contains the entire child node
                child_name = child_node['name']  # Access the 'name' property
                child_type = child_node['labels']
                children.append([child_name, child_type])  # Append the name to the list
            return children
    except Exception as e:
        print("An error occurred:", e)
        return None  # Return None or an empty list in case of error


def update_node_property(driver, database, node_label, existing_property, existing_value, new_property, new_value):
    """
    Update a node's property in the Neo4j database.

    Parameters:
    driver: The Neo4j driver instance to connect to the database.
    database: The name of the database where the node will be updated.
    node_label: The label for the node.
    existing_property: The property to identify the node.
    existing_value: The value of the existing property.
    new_property: The new property to add or update.
    new_value: The value of the new property.
    """
    # Define the query to update the node with the provided label and property
    query = (
        f"MATCH (n:{node_label} {{{existing_property}: $existing_value}}) "
        f"SET n.{new_property} = $new_value "
        "RETURN n"
    )

    # Execute the query using the driver
    try:
        with driver.session(database=database) as session:
            result = session.run(query, existing_value=existing_value, new_value=new_value)
            print(f"Added new propeorty: {new_property} -> {new_value} " )
    except Exception as e:
        print("An error occurred:", e)


def update_relationship_type(driver, database, node_label_1, property_key_1, property_value_1,
                              node_label_2, property_key_2, property_value_2, 
                              old_relationship_type, new_relationship_type):
    """
    Update the type of a relationship between two nodes in the Neo4j database.

    Parameters:
    driver: The Neo4j driver instance to connect to the database.
    database: The name of the database where the relationship will be updated.
    node_label_1: The label for the first node.
    property_key_1: The property key to identify the first node.
    property_value_1: The value of the property for the first node.
    node_label_2: The label for the second node.
    property_key_2: The property key to identify the second node.
    property_value_2: The value of the property for the second node.
    old_relationship_type: The existing type of the relationship to be changed.
    new_relationship_type: The new type of the relationship.
    """
    # Define the query to find the existing relationship and create a new one
    query = (
        f"MATCH (a:{node_label_1} {{{property_key_1}: $property_value_1}})-[r:{old_relationship_type}]->(b:{node_label_2} {{{property_key_2}: $property_value_2}}) "
        f"CREATE (a)-[newRel:{new_relationship_type}]->(b) "
        "SET newRel += properties(r) "  # Correctly copy properties from r to newRel
        "DELETE r "
        "RETURN newRel"
    )

    # Execute the query using the driver
    try:
        with driver.session(database=database) as session:
            result = session.run(query, property_value_1=property_value_1, property_value_2=property_value_2)
            print(f"Changed relationship from {old_relationship_type} to {new_relationship_type}.")
    except Exception as e:
        print("An error occurred:", e)





def delete_node(driver, database, node_name, node_label):
    """
    Delete a Person node and its relationships from the Neo4j database.

    Parameters:
    driver: The Neo4j driver instance to connect to the database.
    database: The name of the database where the node will be deleted.
    person_name: The name of the person to be deleted (e.g., Alice).
    person_label: The label for the node (default is "Person").
    """
    # Define the query to match and delete the person node
    query = f"""
    MATCH (p:{node_label} {{name: $node_name}})
    DETACH DELETE p
    """
    try: 
    # Execute the query using the driver
        with driver.session(database=database) as session:
            session.run(query, node_name=node_name)
            print(f"Person '{node_name}' and their relationships have been deleted.")
    except Exception as e:
        print("An error occurred:", e)

def find_path_nodes(driver, database, source_name, source_type, source_module, destination_name, destination_type, destination_module):
    query = f"""
        MATCH path = (: {source_type} {{name: $source_name, belongs: $source_module}})-[*]->(: {destination_type} {{name: $destination_name, belongs: $destination_module}})
        RETURN [node IN nodes(path) | node.name] AS node_names, [node IN nodes(path) | node.label] AS node_labels
        """
    try:
        with driver.session(database=database) as session:
            print(f"{query}\n")
            result = session.run(query, 
                                 source_name=source_name, 
                                 source_module=source_module, 
                                 destination_name=destination_name, 
                                 destination_module=destination_module)
            return result.single()
    except Exception as e:
        print("An error occurred while finding the path:", e)    



def find_path(source_name, source_type, source_module, destination_name, destination_type, destination_module):
    with GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4jgraph")) as driver:
        driver.verify_connectivity()
        print("Connection established.")
        database = "neo4j"
        result = find_path_nodes(driver, database, source_name='interfaces', source_type='module', source_module='interfaces', destination_name='destination', destination_type='leaf', destination_module='network')

        path_node_names = result["node_names"]
        path_node_labels = result["node_labels"]

        # print(path_node_names, path_node_labels)
        path = ""
        for i in range(len(path_node_labels)):  # Corrected loop
            if path_node_labels[i] == 'list':
                path += f"{path_node_names[i]}[]"
            else:
                path += f"{path_node_names[i]}"
            
            # Optionally, add a separator if needed
            if i < len(path_node_labels) - 1:  # Check to avoid adding after the last element
                path += "\\"  # Use " \ " as a separator between nodes

        print(path)
        driver.close()



def delete_all_nodes_and_relationships(driver, database):
    query = "MATCH (n) DETACH DELETE n"
    try:
        with driver.session(database=database) as session:
            session.run(query)
            print("All nodes and relationships deleted.")
    except Exception as e:
        print("An error occurred while deleting nodes and relationships:", e)

uses_list = []
prefix_dict = {}
all_includes = {}
def parse_modules_containers(filepath, parent_label, parent_name):
    belongs = 0

    """
    This module connects to the database
    Parses each line at a time
    while parsing it creates nodes and relations among the nodes that belong to a YANG file
    """
    with GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4jgraph")) as driver:
        driver.verify_connectivity()
        print("Connection established.")
        database = "neo4j"
        
        stack_braces = [] # matching braces stack
        module_names = [] # noting the name of each node, useful to trace back the nodes in heirachical situation
        module_types = [] # noting the type of each node, useful to trace back the nodes in heirachical situation 
        
        current_module = [] # noting the module names, same as the above

        
               
        with open(filepath, 'r') as file:
            lines = file.readlines() # all lines
            for line in lines: # read one line at a time
                line = line.strip()
                if(line == ''):
				# If its an empty line, do nothing
                    continue
                # print(node_stack)
                line_words = line.split()
                if(parent_name == None):
                    # This needs to be changed, this is basically designed to check if the graph is empty
                    # But this doesn't serve the purpose as we are building the graphs indiviadually and then connecting them
                    if(len(line_words) >= 2 and line_words[0] == 'module' and line_words[-1] == '{'):
                        # If it is "module" then create a node of type module and update the necessary stacks and lists.
                        parent_label = 'module'
                        module_name = line_words[1]
                        create_node(driver, database, node_label = 'module', node_name = line_words[1], node_datatype = 'None', node_parent_name = 'None', node_parent_type = 'None', node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        stack_braces.append('{') # Add a '{' to the stack_braces stack
                        module_names.append(line_words[1]) # Add name of the node to the mudule_names stack
                        module_types.append('module') # Add type of the node to the mudule_types stack
                        parent_name = line_words[1] # Save the current node information so that the next node uses it to create relationship
                        current_module.append(module_name) # Save the current module name to the stack, this functionality doens't require a stack, a simple variable would suffice.
                    elif(len(line_words) >= 2 and line_words[0] == 'submodule' and line_words[-1] == '{'):
                        # If it is a submodule, create a node.
                        # Later part of the program will create a relationship among the parent module and submodule
                        parent_label = 'module'
                        module_name = line_words[1]
                        create_node(driver, database, node_label = 'submodule', node_name = line_words[1], node_datatype = 'None', node_parent_name = 'None', node_parent_type = 'None', node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('submodule')
                        parent_name = line_words[1]
                        current_module.append(module_name)
                        
                else: # if the graph is not empty
                    if(len(line_words) >= 2 and line_words[0] == 'module' and line_words[-1] == '{'):
                        # If the graph is not empty and if it is a module to be created then create it
                        # Create a module with the given name and create a relationship between the previous node and the new module node
                        # This is actually not required as there will not be a case where the graph is not empty and a new module is to be created.
                        module_name = line_words[1]
                        create_node(driver, database, node_label = 'module', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'module', child_name = line_words[1], relationship_type = 'MODULE', belongs=module_name)
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('module')
                        parent_name = line_words[1]
                        parent_label = "module"
                        current_module.append(module_name)
                        
                    elif(len(line_words) >= 2 and line_words[0] == 'import' and line_words[-1] == '{'):
                        # If the statement is a import statement:
                        # There is no need to create a node, as we are not displaying the node in the graph.
                        # this works if the import statement is in multiple lines
                        # If you want to create a node and create a relationship then, uncomment the below code and put it at the end of the current block
                        # create_node(driver, database, node_label = 'import', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        # create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'import', child_name = line_words[1], relationship_type = 'HAS_IMPORT', belongs=module_name)
                        stack_braces.append('{')	
                        module_names.append(line_words[1].strip('"'))
                        module_types.append('import')
                        parent_name = line_words[1].strip('"')
                        parent_label = 'import'
                        current_module.append(module_name)
                        # parse_modules_containers(f'{line_words[1]}.yang', parent_label, parent_name)
                        # node_stack.append(1)
                    elif(len(line_words) >= 2 and line_words[0] == 'import' and line_words[-1] == '}'):
                        # Same as the above but this works if the import statement is in single line
                        match = re.match(r'import\s+([a-zA-Z0-9\-\.]+)\s*\{\s*prefix\s+([a-zA-Z0-9\-]+);?\s*\}', line)
                        import_name = line_words[1]
                        prefix_n = line_words[4].strip(';')
                        prefix_n = prefix_n.strip('"')
                        prefix_dict[prefix_n] = import_name
                        # create_node(driver, database, node_label = 'import', node_name = import_name, node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        # create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'import', child_name = import_name, relationship_type = 'HAS_IMPORT', belongs=module_name)
                    elif(len(line_words) >= 2 and line_words[0] == 'container' and line_words[-1] == '{'):
                        # If the statement is a "container", then create a node and then create a relationship with the previous node
                        # The previous node information is in parent name and parent label variables
                        create_node(driver, database, node_label = 'container', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'container', child_name = line_words[1], relationship_type = 'Container', belongs=module_name)
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('container')
                        current_module.append(module_name)
                        parent_name = line_words[1]
                        parent_label = 'container'
                        # node_stack.append(3)
                    elif(len(line_words) >= 2 and line_words[0] == 'leaf' and line_words[-1] == '{'):
                        # If the statement is a "leaf", then create a node and then create a relationship with the previous node
                        # The previous node information is in parent name and parent label variables
                        create_node(driver, database, node_label = 'leaf', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'leaf', child_name = line_words[1], relationship_type = 'LEAF', belongs=module_name)
                        rel_name = parent_name
                        rel_label = parent_label
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('leaf')
                        current_module.append(module_name)
                        parent_name = line_words[1]
                        parent_label = 'leaf'
                        # node_stack.append(4)
                    elif(len(line_words) >= 2 and line_words[0] == 'notification' and line_words[-1] == '{'):
                        # If the statement is a "notification", then create a node and then create a relationship with the previous node
                        # The previous node information is in parent name and parent label variables
                        create_node(driver, database, node_label = 'notification', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'notification', child_name = line_words[1], relationship_type = "NOTIFICATION", belongs=module_name)
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('notification')
                        current_module.append(module_name)
                        parent_name = line_words[1]
                        parent_label = 'notification'
                    elif(len(line_words) >= 2 and line_words[0] == 'list' and line_words[-1] == '{'):
                        create_node(driver, database, node_label = 'list', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'list', child_name = line_words[1], relationship_type = "LIST", belongs=module_name)
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('list')
                        current_module.append(module_name)
                        parent_name = line_words[1]
                        parent_label = 'list'
                    elif(len(line_words) >= 2 and line_words[0] == 'leaf-list' and line_words[-1] == '{'):
                        create_node(driver, database, node_label = 'leaflist', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'leaflist', child_name = line_words[1], relationship_type = "LEAFLIST", belongs=module_name)
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('leaflist')
                        current_module.append(module_name)
                        parent_name = line_words[1]
                        parent_label = 'leaflist'
                    elif(len(line_words) >= 2 and line_words[0] == 'choice' and line_words[-1] == '{'):
                        create_node(driver, database, node_label = 'choice', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'choice', child_name = line_words[1], relationship_type = "CHOICE", belongs=module_name)
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('choice')
                        current_module.append(module_name)
                        parent_name = line_words[1]
                        parent_label = 'choice'
                    elif(len(line_words) >= 2 and line_words[0] == 'case' and line_words[-1] == '{'):
                        create_node(driver, database, node_label = 'case', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'case', child_name = line_words[1], relationship_type = "CASE", belongs=module_name)
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('case')
                        current_module.append(module_name)
                        parent_name = line_words[1]
                        parent_label = 'case'
                    elif(len(line_words) >= 2 and line_words[0] == 'grouping' and line_words[-1] == '{'):
                        create_node(driver, database, node_label = 'grouping', node_name = line_words[1], node_datatype = 'None', node_parent_name = 'None', node_parent_type = 'None', node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('grouping')
                        current_module.append(module_name)
                        parent_name = line_words[1]
                        parent_label = 'grouping'
                        # node_stack.append(2)
                    elif(len(line_words) >= 2 and line_words[0] == 'rpc' and line_words[-1] == '{'):
                        create_node(driver, database, node_label = 'rpc', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'rpc', child_name = line_words[1], relationship_type = "RPC", belongs=module_name)
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('rpc')
                        current_module.append(module_name)
                        parent_name = line_words[1]
                        parent_label = 'rpc'
                    elif(len(line_words) >= 2 and line_words[0] == 'typedef' and line_words[-1] == '{'):
                        create_node(driver, database, node_label = 'typedef', node_name = line_words[1], node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'typedef', child_name = line_words[1], relationship_type = "TYPEDEF", belongs=module_name)
                        rel_label = 'typedef'
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('typedef')
                        current_module.append(module_name)
                        parent_name = line_words[1]
                        parent_label = 'typedef'
                    elif(len(line_words) >= 2 and line_words[0] == 'belongs-to' and line_words[-1] == '{'):
                        module_name = line_words[1]
                        stack_braces.append('{')
                        module_names.append('other')
                        module_types.append('other')
                        current_module.append(line_words[1])
                        belongs = 1
                    if(line_words[0] == 'input' and line_words[-1] == '{'):
                        create_node(driver, database, node_label = 'input', node_name = f"input {parent_name}", node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'input', child_name = f"input {parent_name}", relationship_type = "INPUT", belongs=module_name)
                        stack_braces.append('{')
                        module_names.append(f"input {parent_name}")
                        current_module.append(module_name)
                        module_types.append('input')
                        parent_name = f"input {parent_name}"
                        parent_label = 'input'
                    if(line_words[0] == 'output' and line_words[-1] == '{'):
                        create_node(driver, database, node_label = 'output', node_name = f"output {parent_name}", node_datatype = 'None', node_parent_name = parent_name, node_parent_type = parent_label, node_mod_belongs = module_name, node_value = 'None', node_description = 'Empty')
                        create_relationship(driver, database, parent_type = parent_label, parent_name = parent_name, child_type = 'output', child_name = f"output {parent_name}", relationship_type = "OUTPUT", belongs=module_name)
                        stack_braces.append('{')
                        module_names.append(f"output {parent_name}")
                        current_module.append(module_name)
                        module_types.append('output')
                        parent_name = f"output {parent_name}"
                        parent_label = 'output'


                    """ 
                    Uses:
                    groupname is the group that is being used
                    prefix is the module prefix in which the group exists
                    parent label is the parent type
                    parent name is the name of the parent
                    module_name is the name of the module
                    using all this information, create a link in the below format:
                        find all of the nodes that are direct children of the group
                        to all the children create link to the current node 

                    """

                    if(line_words[0] == 'uses'):
                        if(line_words[-1]!='{'):
                            group_name = line_words[1].rstrip(';')
                            temp_uses = []
                            prefix = ''
                            if ':' in group_name:
                                separated_values = group_name.split(':')
                                prefix = separated_values[0]
                                group_name = separated_values[1]
                            temp_uses.append([module_name, parent_name, parent_label, prefix, group_name])
                            # print(temp_uses)
                            uses_list.append(temp_uses)
                        elif(line_words[-1]=='{'):
                            stack_braces.append('{')
                            module_names.append('other')
                            module_types.append('other')
                            current_module.append(module_name)

                    
                    if(line_words[0] == 'type' and line.endswith(';')):
                        type_name = line_words[1].rstrip(';')
                        type_name = type_name.replace('-', '_')
                        type_name = type_name.replace(':', '_')
                        type_name = type_name.replace('.', '_')
                        type_name = type_name.replace('{', "")
                        new_relation = type_name.upper()
                        update_node_property(driver, database, node_label = parent_label, existing_property = 'name', existing_value = parent_name, new_property = 'datatype', new_value = type_name)
                        if(rel_label != 'typedef' and (rel_label != 'ENUM' or rel_label != 'enum')):
                            update_relationship_type(driver, database, node_label_1 = rel_label, property_key_1= 'name', property_value_1 = rel_name, node_label_2='leaf', property_key_2 = 'name', property_value_2 = parent_name, old_relationship_type='LEAF', new_relationship_type=new_relation)
                    elif(len(line_words) >=2 and line_words[0] == 'type' and line_words[1] == 'enumeration'):
                        update_node_property(driver, database, node_label = parent_label, existing_property = 'name', existing_value = parent_name, new_property = 'datatype', new_value = 'enum')
                        new_relation = 'ENUM'
                        # update_relationship_type(driver, database, node_label_1 = rel_label, property_key_1= 'name', property_value_1 = rel_name, node_label_2='leaf', property_key_2 = 'name', property_value_2 = parent_name, old_relationship_type='LEAF', new_relationship_type=new_relation)
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('enum')
                        current_module.append(module_name)
                    elif(line_words[0] == 'type' and line.endswith('{')):
                        type_name = line_words[1]
                        type_name = type_name.replace('-', '_')
                        type_name = type_name.replace(':', '_')
                        type_name = type_name.replace('.', '_')
                        type_name = type_name.replace('{', "")
                        new_relation = type_name.upper()
                        update_node_property(driver, database, node_label = parent_label, existing_property = 'name', existing_value = parent_name, new_property = 'datatype', new_value = type_name)
                        if(rel_label != 'typedef' and (rel_label != 'ENUM' or rel_label != 'enum')):
                            update_relationship_type(driver, database, node_label_1 = rel_label, property_key_1= 'name', property_value_1 = rel_name, node_label_2='leaf', property_key_2 = 'name', property_value_2 = parent_name, old_relationship_type='LEAF', new_relationship_type=new_relation)
                        stack_braces.append('{')
                        module_names.append(line_words[1])
                        module_types.append('type')
                        current_module.append(module_name)
                    

                    if(line_words[0] == 'key'):
                        type_name = line_words[1].rstrip(';')
                        type_name = type_name.strip('"')
                        update_node_property(driver, database, node_label = parent_label, existing_property = 'name', existing_value = parent_name, new_property = 'key', new_value = type_name)
                    
                    if(line_words[0] == 'prefix'):
                        prefix_name = line_words[1].rstrip(';')
                        prefix_name = prefix_name.strip('"')
                        prefix_dict[prefix_name] = module_names[-1]
                        # print(f"{line} ----------- {prefix_name} --->>>> {module_names[-1]}")
                        # time.sleep(10)
                        if(belongs == 1):
                            belongs = 0
                        else:
                            update_node_property(driver, database, node_label = parent_label, existing_property = 'name', existing_value = parent_name, new_property = 'prefix', new_value = prefix_name)
                    
                    if(line_words[0] == 'include' and line_words[1].endswith(';')):
                        prefix_name = line_words[1].rstrip(';')
                        prefix_name = prefix_name.strip('"')
                        all_includes[prefix_name] = module_name

                    if(len(line_words) >= 2 and line_words[0] != 'container' and line_words[0] != 'import' and line_words[0] != 'module' and line_words[0] != 'leaf' and line_words[0] != 'grouping' and line_words[0] != 'list' and line_words[0] != 'leaf-list' and line_words[0]!= 'rpc' and line_words[0]!= 'input' and line_words[0]!= 'notification' and line_words[0]!= 'output' and line_words[0]!= 'typedef' and line_words[1]!= 'enumeration' and line_words[0]!= 'type' and line_words[0]!= 'belongs-to' and line_words[0]!= 'uses' and line_words[0]!= 'choice' and line_words[0]!= 'case' and line_words[-1] == '{'):
                        print("Not any but appending")
                        stack_braces.append('{')
                        module_names.append('other')
                        module_types.append('other')
                        current_module.append(module_name)
                        
                    if(line == '}'):
                        # print("Need to pop")
                        # print('module_names: ', module_names)
                        # print('stack_braces: ',stack_braces)
                        # print('module_types', module_types)
                        # print('current_module:', current_module)
                        if(len(module_names) == len(stack_braces)):
                            stack_braces.pop()
                            module_names.pop()
                            module_types.pop()
                            current_module.pop()
                            # node_stack.pop()
                            if(len(module_names)>0):
                                parent_name = module_names[-1]
                                parent_label = module_types[-1]
                                module_name = current_module[-1]
                            else:
                                return
                        else:
                            if(len(stack_braces)>0):
                                stack_braces.pop()
                                module_names.pop()
                                module_types.pop()
                                current_module.pop()
                                # node_stack.pop()
                            else:
                                return
                    # print("\n Line: ---->", line, line_words)
                    # print("\n")
                    # print(f"Stack Braces: {stack_braces} \n module name: {module_names}\n module Types: {module_types} \n current Module: {current_module}")
                    # time.sleep(10)
        # delete_all_nodes_and_relationships(driver, database)
        driver.close()
    print("Done")


# First way of running the program:
# get all the YANG file names that are related to a given YANG and run them all.
count = 0
all_yangs = acy.send_related_yangs('openconfig-network-instance.yang')
root = 'C:\\Users\\2799361\\Desktop\\GenAI\\Yang_to_Graph\\all_openconfig_yangs\\'

for file_name in all_yangs:
    full_path = os.path.join(root, file_name)
    parse_modules_containers(full_path, None, None)
    count += 1
# First way ends here




# Second way of running the code
# Just run the code for all the YANG files present in a folder and then create links

def get_yang_files(directory):
    """
    Get a list of all .yang files in the specified directory.

    :param directory: str - Path to the directory to search for .yang files.
    :return: list - A list of .yang file paths.
    """
    yang_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Check if the file ends with .yang
            if filename.endswith('.yang'):
                # Join the directory and file name to get the full path
                full_path = os.path.join(root, filename)
                yang_files.append(full_path)
                # yang_files.append(filename)

    return yang_files

# Specify the directory you want to search
# directory_path = 'C:\\Users\\2799361\\Desktop\\GenAI\\Yang_to_Graph\\all_oran\\'
# directory_path = 'C:\\Users\\2799361\\Desktop\\GenAI\\Yang_to_Graph\\New folder\\'

# Get the list of .yang files
# yang_files = get_yang_files(directory_path)

# for yang_file in yang_files:
#     # print(yang_file)
    # parse_modules_containers(yang_file, None, None)
# Second way ends here




# Users List has data in the given format:
# [module_name, parent_name, parent_label, prefix, group_name]


# Prefix dictionary has all the yang prefixes
# This is used to create links between nodes, if the grouping is from other modules.



print("Uses List: ", uses_list)
print("Prefix Dictionary: ", prefix_dict)

# time.sleep(20)

def create_group_relationship(driver, database, pparent_type, pparent_name, parent_belongs, child_name, relationship_type):
    """
    Create a relationship between two nodes in the Neo4j database, preventing duplicates.

    Parameters:
    driver: The Neo4j driver instance to connect to the database.
    database: The name of the database where the relationship will be created.
    parent_type: The label/type of the parent node (e.g., Person).
    parent_name: The name of the parent node (e.g., Alice).
    child_type: The label/type of the child node (e.g., Person).
    child_name: The name of the child node (e.g., Bob).
    relationship_type: The type of relationship to create (e.g., KNOWS).
    """
    # Define the query to match the two nodes and create a relationship if it doesn't exist
    query = f"""
    MATCH (p1:{pparent_type} {{name: $pparent_name, belongs: $parent_belongs}}), (p2 {{name: $child_name, parent_type: 'grouping'}})
    MERGE (p1)-[:{relationship_type}]->(p2)
    """

    # Execute the query in the provided database context
    try:
        with driver.session(database=database) as session:
            print(f"{query}\n")
            session.run(query, pparent_type = pparent_type, pparent_name =  pparent_name, parent_belongs = parent_belongs,  child_name=child_name)
            print(f"Relationship '{relationship_type}' created between {pparent_name} and {child_name}. \n \n")
    except Exception as e:
        print("An error occurred while creating the relationship:", e)



def create_include_relationship(driver, database, parent_type, parent_name, child_type, child_name, relationship_type):
    """
    Create a relationship between two nodes in the Neo4j database, preventing duplicates.

    Parameters:

    """
    # Define the query to match the two nodes and create a relationship if it doesn't exist
    query = f"""
    MATCH (p1:{parent_type} {{name: $parent_name}}), (p2:{child_type} {{name: $child_name}})
    MERGE (p1)-[:{relationship_type}]->(p2)
    """

    # Execute the query in the provided database context
    try:
        with driver.session(database=database) as session:
            print(f"{query}\n")
            session.run(query, parent_type = parent_type, parent_name =  parent_name, child_type = child_type,  child_name=child_name)
            print(f"Relationship '{relationship_type}' created between {parent_name} and {child_name}. \n \n")
    except Exception as e:
        print("An error occurred while creating the relationship:", e)


with GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "neo4jgraph")) as driver:
    driver.verify_connectivity()
    print("Connection established.")
    database = "neo4j"
    print(uses_list)
    for grouping in uses_list:
        for groups in grouping:
            parent_belongs = groups[0]
            pparent_name = groups[1]
            pparent_type = groups[2]
            grouping_name = groups[4]
            if(groups[3] == ''):
                child_belongs = groups[0]
            else:
                child_belongs = prefix_dict[groups[3]]
                
            # print(f"parent_name: {parent_name} \n parent_type: {parent_type} \n parent_module: {module_name} \n child")
            children =  get_group_children(driver, database, node_label='grouping', group_name=grouping_name, mod_belongs=child_belongs)
            # for pairs in range(len(children)):
            #     print(children[pairs][0])
            for pairs in range(len(children)):
                create_group_relationship(driver, database, pparent_type = pparent_type, pparent_name=pparent_name, parent_belongs=parent_belongs, child_name=children[pairs][0], relationship_type="CHILD")
            # print(children)
    for child_name, parent_name in all_includes.items():
        create_include_relationship(driver, database, parent_type = 'module', parent_name = parent_name, child_type = 'submodule', child_name = child_name, relationship_type = "SUBMODULE")


    # find_path(source_name='interfaces', source_type='module', source_module='interfaces', destination_name='destination', destination_type='leaf', destination_module='network')

    
    driver.close()



