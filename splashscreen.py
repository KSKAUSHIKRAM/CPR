from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from screen import *
class Splashscreen(BoxLayout):
    def __init__(self,**kwargs):
        super().__init__(orientation='vertical', spacing=10,size=(0.75,1), padding=[800,20,30,200], **kwargs)
        self.input=TextInput(hint_text="Enter the location",size_hint=(None,None),width=300,height=40)
        submit_btn=Button(text="Submit",size_hint=(None,None),width=300,height=40)
        submit_btn.bind(on_press=self.on_submit)
        self.add_widget(self.input)
        self.add_widget(submit_btn)

    def on_submit(self,instance):
        location=self.input.text
        print("You entered this:",self.input.text)
        self.Myapp.second_scree()
class Myapp(App):
    def build(self):
        return Splashscreen()
    def second_screen(self):
        self.add_widget(AACApp())
if __name__=='__main__':
    Myapp().run()