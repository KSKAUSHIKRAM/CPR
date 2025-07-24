from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from main import *
class Splashscreen(Screen):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        layout=BoxLayout(orientation='vertical', spacing=10,size=(0.75,1), padding=[800,20,30,200])
        self.input=TextInput(hint_text="Enter the location",size_hint=(None,None),width=300,height=40)
        submit_btn=Button(text="Submit",size_hint=(None,None),width=300,height=40)
        submit_btn.bind(on_press=self.on_submit)
        layout.add_widget(self.input)
        layout.add_widget(submit_btn)
        self.add_widget(layout)


    def on_submit(self,instance):
        self.manager.current='aac'
class Myapp(App):
    def build(self):
        sm=ScreenManager()
        sm.add_widget(Splashscreen(name='splash'))
        sm.add_widget(AACApp(name='aac'))
        return sm

if __name__=='__main__':
    Myapp().run()