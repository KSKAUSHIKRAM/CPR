from Model.Database import Database
class Database_helper:
    def __init__(self):
        self.obj=Database()
    def insert(self,text,location):
        self.obj.insert(text,location)
    def recommend(self,location):
        self.obj.recommend(location)