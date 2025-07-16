import re, string, pandas as pd
from Data.Raw import raw_data


def remove_links(text):
    links_pattern = re.compile(
        r'https?://\S+|'
        r'www\.\S+|'
        r't\.me/\S+|'
        r'\b\S+\.(com|ru|org|net|io|ly|info|site|dev|app)\S*|'
        r'vk\.com/\S+|'
        r'@\w+|'
        r'#\w+|',
        flags=re.UNICODE
    )
    return links_pattern.sub('', text)

def remove_emojis(text):
    emojis_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"
        u"\u3030"
        "]+",
        flags=re.UNICODE
    )
    return emojis_pattern.sub('', text)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = remove_links(text)
    text = remove_emojis(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    text = re.sub(r'\s+', ' ', text)
    return text

def clean_data_frame(data_frame):
    data_frame['text'] = data_frame['text'].astype(str).str.strip()
    data_frame = data_frame[data_frame['text'] != ""]
    data_frame = data_frame.reset_index(drop=True)
    return data_frame

def texts_to_df(texts):
    cleaned_texts = [clean_text(text) for text in texts]
    data_frame = clean_data_frame(pd.DataFrame(cleaned_texts, columns=["text"]))
    return data_frame

# test = raw_data.load_text_from_json("/Users/ivan/PycharmProjects/PostsFilter/Data1/posts.json")
# df = texts_to_df(test)
# print(df.head(60))
