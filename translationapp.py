import os
import tempfile
import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from sentence_transformers import SentenceTransformer, util
import whisper
from gtts import gTTS
import gradio as gr
import google.generativeai as genai 

# Add ffmpeg to PATH for audio processing


#  Load Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))
# 1. spaCy language detector
@Language.factory("language_detector")
def create_lang_detector(nlp, name):
    return LanguageDetector()

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector", last=True)


# 2. M2M100 translation model
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
translation_model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# 3. Whisper
whisper_model = whisper.load_model("small")

# 4. Sentence similarity model
similarity_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# 5. Language map
lang_map = {
    "af": "Afrikaans","ast": "Asturian",
    "be": "Belarusian","bg": "Bulgarian","bn": "Bengali",
    "bs": "Bosnian","ca": "Catalan","cs": "Czech","da": "Danish",
    "de": "German","el": "Greek","en": "English","es": "Spanish",
    "fi": "Finnish","fr": "French","fy": "Western Frisian",
    "gl": "Galician","ha": "Hausa",
    "hi": "Hindi","hr": "Croatian","hu": "Hungarian","hy": "Armenian",
    "id": "Indonesian","it": "Italian","tr": "Turkish","uk": "Ukrainian",
    "ja": "Japanese","ko": "Korean",
    "lt": "Lithuanian","lv": "Latvian","mk": "Macedonian",
    "mr": "Marathi","ms": "Malay","nl": "Dutch","no": "Norwegian",
    "pl": "Polish","pt": "Portuguese","ro": "Romanian","ru": "Russian",
    "sk": "Slovak","sl": "Slovenian","oc": "Occitan","ps": "Pashto",
    "sq": "Albanian","ml": "Malayalam","lb": "Luxembourgish","ka": "Georgian","ne": "Nepali",
    "sv": "Swedish","th": "Thai","tl": "Tagalog","tn": "Tswana",
    "ur": "Urdu","ar": "Arabic","zh": "Chinese","ceb": "Cebuano","fa": "Persian","et": "Estonian","jv": "Javanese","lo": "Lao"
}
dropdown_choices = [f"{name} ({code})" for code, name in lang_map.items()]

# 6. Helpers
def get_lang_code(selected):
    return selected.split("(")[-1].replace(")", "").strip()

def detect_language_spacy(text):
    doc = nlp(text)
    lang = doc._.language["language"]
    if lang.startswith("zh"): lang = "zh"
    if lang in ["fa", "ar"]: return "ur"
    return lang

def translate(text, target_lang_choice):
    try:
        if not text.strip():
            return "Please enter some text.", "N/A"
        
        target_lang = get_lang_code(target_lang_choice)
        src_lang = detect_language_spacy(text)

        if src_lang not in lang_map or target_lang not in lang_map:
            return f"Unsupported: {src_lang} → {target_lang}", lang_map.get(src_lang, src_lang)
        
        tokenizer.src_lang = src_lang
        sentences = [sent.text.strip() for sent in nlp(text).sents if sent.text.strip()]
        translations = []
        for sent in sentences:
            tokenized = tokenizer(sent, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            outputs = translation_model.generate(
                **tokenized,
                max_length=1024,
                num_beams=3,
                forced_bos_token_id=tokenizer.get_lang_id(target_lang)
            )
            decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in outputs]
            translations.extend(decoded)
        return " ".join(translations), f"{lang_map.get(src_lang, src_lang)} ({src_lang})"
    except Exception as e:
        return f"Error: {str(e)}", "Error"

def semantic_similarity(text1, text2):
    if not text1.strip() or not text2.strip():
        return "Please enter both texts."
    embeddings = similarity_model.encode([text1, text2], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return f"{score*100:.2f}% similar"

def speech_to_text(audio_file):
    try:
        result = whisper_model.transcribe(audio_file)
        return result["text"], f"Detected: {result['language']}"
    except Exception as e:
        return f"Error: {e}", "Error"


def text_to_speech(text, lang="en"):
    if not text.strip():
        return None
    tts = gTTS(text, lang=lang)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    return temp_file.name

# Gemini Chat Function
def gemini_chat(user_input, history):
    if not user_input.strip():
        return history

    try:
        model = genai.GenerativeModel("gemini-2.5-pro")
        chat = model.start_chat(history=[])
        response = chat.send_message(user_input)
        reply = response.text

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        return history
    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        return history

# 7. Gradio UI
with gr.Blocks(css="""
    .bottom-nav { 
        position: fixed; 
        bottom: 0; 
        left: 0; 
        width: 100%; 
        display: flex; 
        justify-content: space-around; 
        background: #f5f5f5; 
        border-top: 1px solid #ccc; 
        padding: 10px 0; 
    }
    .bottom-nav button {
        flex: 1;
        margin: 0 5px;
        border-radius: 12px;
    }
""") as iface:
    gr.Markdown("LANGUAGE TRANSLATER & GEMINI CHATBOT FOR SEARCH")

    # Pages
    with gr.Group(visible=True) as translator_page:
        with gr.Column():
            gr.Markdown("Translator")
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙 Speak")
            text_input = gr.Textbox(label="Transcribed / Input text")
            detected_lang = gr.Textbox(label="Detected language")

            target_lang = gr.Dropdown(choices=dropdown_choices, label="Target language", value="English (en)")
            translated_text = gr.Textbox(label=" Translated text")
            audio_output = gr.Audio(label="Listen", type="filepath")

            audio_btn = gr.Button("Convert Speech to Text")
            audio_btn.click(fn=speech_to_text, inputs=audio_input, outputs=[text_input, detected_lang])

            translate_btn = gr.Button("Translate")
            translate_btn.click(fn=translate, inputs=[text_input, target_lang], outputs=[translated_text, detected_lang])

            tts_btn = gr.Button("Speak Translation")
            tts_btn.click(fn=text_to_speech, inputs=translated_text, outputs=audio_output)

    with gr.Group(visible=False) as compare_page:
        with gr.Column():
            gr.Markdown(" Compare Texts")
            text1 = gr.Textbox(label="Text 1")
            text2 = gr.Textbox(label="Text 2")
            similarity_output = gr.Textbox(label="Similarity Score")
            compare_btn = gr.Button("Compare")
            compare_btn.click(fn=semantic_similarity, inputs=[text1, text2], outputs=similarity_output)

    with gr.Group(visible=False) as chat_page:
        with gr.Column():
            gr.Markdown(" Gemini Chatbot")
            gemini_chatbot = gr.Chatbot(label="Gemini Chat", type="messages")
            gemini_msg = gr.Textbox(label="Type your message to Gemini")
            gemini_clear = gr.Button("Clear Chat")

            gemini_msg.submit(gemini_chat, [gemini_msg, gemini_chatbot], gemini_chatbot)
            gemini_clear.click(lambda: [], None, gemini_chatbot, queue=False)

    # Bottom Navigation 
    with gr.Row(elem_classes="bottom-nav"):
        nav_translator = gr.Button("Translator")
        nav_compare = gr.Button(" Compare")
        nav_chat = gr.Button(" Search Information About language")

    # Switch Pages
    def show_translator(): return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    def show_compare(): return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    def show_chat(): return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)

    nav_translator.click(show_translator, None, [translator_page, compare_page, chat_page])
    nav_compare.click(show_compare, None, [translator_page, compare_page, chat_page])
    nav_chat.click(show_chat, None, [translator_page, compare_page, chat_page])

iface.launch()