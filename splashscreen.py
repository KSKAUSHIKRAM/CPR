from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.floatlayout import FloatLayout
from main import *

class Splashscreen(Screen):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        root=FloatLayout()
        layout=BoxLayout(orientation='vertical', spacing=10,size_hint=(0.3,0.2), pos_hint={'center_x':0.5,'center_y':0.5})
        self.input=TextInput(hint_text="Enter the location",size_hint=(None,None),width=300,height=40)
        submit_btn=Button(text="Submit",size_hint=(None,None),width=300,height=40)
        submit_btn.bind(on_press=self.on_submit)
        layout.add_widget(self.input)
        layout.add_widget(submit_btn)
        #self.add_widget(layout)
        root.add_widget(layout)
        self.add_widget(root)


    def on_submit(self,instance):
        self.manager.current='aac'
class Myapp(App):
    def build(self):
        sm=ScreenManager()
        sm.add_widget(Splashscreen(name='splash'))
        sm.add_widget(AACApp(name='aac'))
        return sm
    def exit_app(self, instance):
        App.get_running_app().stop()
if __name__=='__main__':
    Myapp().run()