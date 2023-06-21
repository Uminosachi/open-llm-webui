from transformers import pipeline
import pysbd
from tqdm import tqdm

seg_en = pysbd.Segmenter(language="en", clean=False)
seg_ja = pysbd.Segmenter(language="ja", clean=False)

translator_en_ja = None
translator_ja_en = None

def load_translator():
    global translator_en_ja, translator_ja_en
    # translator_en_ja = pipeline("translation", model="staka/fugumt-en-ja")
    # translator_ja_en = pipeline("translation", model="staka/fugumt-ja-en")
    # translator_en_ja = pipeline("translation", model="Helsinki-NLP/opus-mt-en-jap")
    translator_ja_en = pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")
    translator_en_ja = pipeline("translation",
                                model="facebook/mbart-large-50-one-to-many-mmt",
                                src_lang="en_XX", tgt_lang="ja_XX")
    # translator_ja_en = pipeline("translation",
    #                             model="facebook/mbart-large-50-one-to-many-mmt",
    #                             src_lang="ja_XX", tgt_lang="en_XX")

def translate(text, src_lang, tgt_lang):
    global translator_en_ja, translator_ja_en
    if translator_en_ja is None or translator_ja_en is None:
        load_translator()

    if src_lang == "en" and tgt_lang == "ja":
        seg_text = seg_en.segment(text)
        translated_text = []
        for seg in tqdm(seg_text):
            translated_text.append(translator_en_ja(seg)[0]["translation_text"].strip())
        return "".join(translated_text)
    elif src_lang == "ja" and tgt_lang == "en":
        seg_text = seg_ja.segment(text)
        translated_text = []
        for seg in tqdm(seg_text):
            translated_text.append(translator_ja_en(seg)[0]["translation_text"].strip())
        return "".join(translated_text)
    else:
        raise NotImplementedError
