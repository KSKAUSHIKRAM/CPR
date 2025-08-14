from happytransformer import HappyTextToText

happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")

def autocorrect_text(text):
    result = happy_tt.generate_text(f"grammar: {text}")
    return result.text


autocorrect_text("i wann wate")