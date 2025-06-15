import streamlit as st
from PIL import Image
from rag_pipeline import load_and_translate_documents, create_vectorstore, retrieve_relevant_chunks
from model_utils import load_model, generate_response
from llava_utils import load_llava_model, analyze_image

st.set_page_config(page_title="Dijital Pazarlama Chatbot'u", layout="wide")

st.image("FDEEP.AI.png", width=300)
st.title("📈 Dijital Pazarlama ve Dijital Reklamcılık Chatbot'u 🤖 ")
st.markdown("Kişisel dijital pazarlama asistanınız 🤖 ")

# Belge klasörü yolu
PDF_FOLDER = r"C:\Users\feyza\OneDrive\Masaüstü\teknofest_chatbot\dijitalpazarlama_reklamkaynakları"

# Belgeleri yükle ve vektör oluştur
with st.spinner("Belgeler işleniyor..."):
    documents = load_and_translate_documents(PDF_FOLDER)
    vectorstore = create_vectorstore(documents)

# Model seçimi
model_key = st.selectbox("Kullanmak istediğiniz modeli seçin:", ["mistral", "llama3.1.8", "qwen3", "gemma3", "deepseek-r1"])
model_name = load_model(model_key)

# 🧠 Chat Arayüzü
st.subheader("💬 Soru-Cevap (Chatbot)")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Dijital pazarlama ve dijital reklamcılık ile ilgili sorunuzu yazın:")

if st.button("Gönder"):
    if not user_input.strip():
        st.warning("Lütfen bir soru yazın.")
    else:
        relevant_docs = retrieve_relevant_chunks(vectorstore, user_input)
        context = "\n".join(relevant_docs)
        prompt = f"Belgelere dayalı olarak yanıt ver. Cevap Türkçe olsun ve sade bir dille yaz:\n\nKullanıcının sorusu: {user_input}\n\nİlgili belgeler:\n{context}"

        with st.spinner("Yanıt oluşturuluyor..."):
            answer = generate_response(model_name, prompt)

        st.session_state.chat_history.append(("🧑‍💻 Soru", user_input))
        st.session_state.chat_history.append(("🤖 Yanıt", answer))

# Sohbet geçmişini göster
for role, msg in st.session_state.chat_history:
    st.markdown(f"*{role}:* {msg}")

# 🖼️ LLaVA ile Görsel Yorumlama
st.subheader("🖼️ Görsel Yorumlama Asistanı (LLaVA)")

uploaded_image = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])
question = st.text_input("Görselle ilgili ne öğrenmek istiyorsunuz?")

if uploaded_image and question:
    with st.spinner("LLaVA modeli yükleniyor..."):
        processor, llava_model = load_llava_model()

    image = Image.open(uploaded_image).convert("RGB")

    with st.spinner("Görsel analiz ediliyor..."):
        output = analyze_image(processor, llava_model, image, question)
        st.image(image, caption="Yüklenen Görsel", use_column_width=True)
        st.markdown("*Yanıt:*")
        st.write(output)
