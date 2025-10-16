from Model.Database import Database
class Database_helper:
    def __init__(self):
        self.obj=Database()
    def insert(self,text,location):
        self.obj.insert(text,location)
    def matched_loc(self,location):
        sentences=self.obj.location_matched(location)
        return sentences or []
    def unmatched_loc(self,location):
        sentences=self.obj.location_unmatched(location)
        return sentences or []

    def retrive_last_inserted(self,location_tag):
        print("Retrieving last inserted sentence for location:", self.obj.last_sentence(location_tag))
        return self.obj.last_sentence(location_tag)
    def unmatched_time(self,location):
        sentences=self.obj.time_phase_unmatched(location)
        return sentences or []
    
