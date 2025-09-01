from Model.Database import Database
class Database_helper:
    def __init__(self):
        self.obj=Database()
    def insert(self,text,location):
        self.obj.insert(text,location)
    def recommend(self,location):
        sentences=self.obj.recommend(location)
        return sentences or []
    def retrive_last_inserted(self,location_tag):
        return self.obj.last_sentence(location_tag)
    
