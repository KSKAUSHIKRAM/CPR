
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
class VirtualKeyboard(GridLayout):
    def __init__(self, target_input, **kwargs):
        super().__init__(**kwargs)
        self.cols = 10
        self.size_hint = (1, None)
        self.height = 240  # Adjust as needed
        self.spacing = 2
        self.padding = 2
        self.pos_hint = {'x':0, 'y':0}  # Always bottom

        self.target_input = target_input

        keys = list("QWERTYUIOPASDFGHJKLZXCVBNM") + ['SPACE', '←', 'CLR']

        for k in keys:
            btn = Button(
                text=k, font_size='20sp',
                size_hint=(None, None), size=(60, 60),
                background_color=(0.2, 0.2, 0.2, 1), color=(1, 1, 1, 1)
            )
            btn.bind(on_press=self.on_key_press)
            self.add_widget(btn)

    def on_key_press(self, instance):
        current = self.target_input.text
        if instance.text == '←':
            self.target_input.text = current[:-1]
        elif instance.text == 'CLR':
            self.target_input.text = ''
        elif instance.text == 'SPACE':
            self.target_input.text += ' '
        else:
            self.target_input.text += instance.text