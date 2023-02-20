import importlib

def get_class_from_file(filepath: str, class_name: str):
    # Import the module specified by the file path
    module_name = filepath.replace('.py', '').replace('/', '.')
    module = importlib.import_module(module_name)

    # Get the class from the module
    class_ = getattr(module, class_name)
   
    return class_

def objective_function(filepath: str, objective_function: str):
        obj = get_class_from_file(filepath, "objectiveFunction")
        objectiveFunction = obj()
        return getattr(objectiveFunction, objective_function)