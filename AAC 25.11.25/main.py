from kivy.app import App
from kivy.config import Config
from kivy.core.window import Window
from Control.Controller import Execute
from Model.Database import Database

# Window setup
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '600')
Window.title = "CPR"

class Myapp(App):
    def build(self):
        db = Database()
        self.sm = Execute(db)   # ✅ Use only Execute
        return self.sm
        

if __name__ == '__main__':
    Myapp().run()
