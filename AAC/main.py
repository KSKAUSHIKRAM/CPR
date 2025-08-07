from kivy.app import App
from kivy.uix.label import Label
from kivy.config import Config
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1024')
Config.set('graphics', 'height', '600')

from kivy.core.window import Window
from Control.Controller import Execute
#set window size
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from Model.Database import Database
# ...rest of your imports

Window.title="CPR"


class Myapp(App):
    def build(self):
      self.db = Database()
      self.sm = Execute(self.db)  # store reference
      return self.sm
if __name__=='__main__':
    Myapp().run()
