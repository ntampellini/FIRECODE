import os
import shutil

class NewFolderContext:
    '''
    Context manager: creates a new directory and moves into it on entry.
    On exit, moves out of the directory and deletes it if instructed to do so.
     
    '''

    def __init__(self, new_folder_name, delete_after=True):
        self.new_folder_name = new_folder_name
        self.delete_after = delete_after

    def __enter__(self):
        # create working folder and cd into it
        if self.new_folder_name in os.listdir():
            shutil.rmtree(os.path.join(os.getcwd(), self.new_folder_name))
            
        os.mkdir(self.new_folder_name)
        os.chdir(os.path.join(os.getcwd(), self.new_folder_name))

    def __exit__(self, *args):
        # get out of working folder
        os.chdir(os.path.dirname(os.getcwd()))

        # and eventually delete it
        if self.delete_after:
            shutil.rmtree(os.path.join(os.getcwd(), self.new_folder_name))
