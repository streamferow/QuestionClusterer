import json

def load_text_from_json(path):
    with open(path, "r", encoding='utf-8') as file:
        data_from_json = json.load(file)

    texts = []
    for text in data_from_json:
        text = text.get("text")
        texts.append(text)

    return texts





