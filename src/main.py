# Imports
# import needed modules here
import argparse
# Function definitions
# need to fill out the following argument.s 
class defaultParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_core_args()
    def getParser(self):
        return self.parser
    def add_core_args(self):
        
       Core_args= self.parser.add_argument_group("Core Arguments")
       Core_args.add_argument("--isTrain")
       Core_args.add_argument("--fileLocation")
       Core_args.add_argument("")
        
def main():
    """
    SCript should do the following, load in the arguments and then feed into run.py
    run.py should be able to load up a model and the data, as well as the data loaders....
    then afterwards it should train all three models and saving them in a seperate folder, with results. 
    
    
    """
    # Your main code goes here
    # 
    parser=defaultParser().getParser()
    args=parser.parse_args()
    
    

# Other functions can be defined here

# This conditional statement checks if the script is the main program
if __name__ == "__main__":
    main()
