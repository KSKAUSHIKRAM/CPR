from kivy.uix.screenmanager import ScreenManager,FadeTransition
from kivy.core.window import Window
from View.firstscreen import SplashScreen
from Model.Database import Database
from View.AACScreen import AACScreen


def insert(text,location):
    obj=Database()
    obj.insert(text,location)
    
class Execute(ScreenManager):
    def __init__(self,db, **kwargs):
        super().__init__(**kwargs)
        self.transition = FadeTransition(duration=0.5)
        # Add all screens here
        self.add_widget(SplashScreen(name="splash"))
        self.add_widget(AACScreen(name="aac"))
        

        self.current = "splash"
