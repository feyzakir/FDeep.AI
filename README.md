# FDeep.AI: Dijital Pazarlama ve Reklam Asistanı (Teknofest 2025)

## Proje Tanımı
FDeepAI, KOBİ'ler ve bireysel girişimciler için ajanslara olan bağımlılığı azaltmayı hedefleyen, **doğal dil işleme tabanlı, çok modelli bir yapay zeka asistanıdır**. Sistem, dijital pazarlama ve reklamcılık konularında bilgi sunar, kullanıcıdan gelen soruları daha önceden yüklenmiş PDF belgelerine dayalı olarak analiz ederek Türkçe yanıtlar üretir. Özellikle teknik bilgisi sınırlı kullanıcıların bile profesyonel kampanya stratejileri oluşturmasına olanak sağlar. Ayrıca tamamen yerel ortamda çalışan bir yapıya sahip olması sayesinde veri gizliliği %100 güvence altına alınmaktadır. Bu özellik, veri güvenliğine öncelik veren büyük ölçekli şirketler için de önemli bir tercih sebebi oluşturmaktadır.

------------
 
## Kullanılan Teknolojiler ve Mimariler

- **Retrieval-Augmented Generation (RAG)** mimarisi
- **LangChain**: Belge bölme, arama zinciri, RAG yapısı
- **FAISS + SentenceTransformers**: Vektör tabanlı belge arama
- **Ollama + LLaMA.cpp**: Yerel LLM çalıştırma (Mistral, LLaMA, Qwen, Gemma, DeepSeek destekli)
- **Streamlit**: Web arayüzü geliştirme
- **Python** (3.10+), CUDA destekli `torch`, `langchain`, `sentence-transformers`
- - **LLaVA 1.5**: Görselden içerik yorumlama (image captioning, sentiment analysis) (LLaVA ile görsel yorumlama özelliği düzeltilecektir.)
 
------------
 
## Kurulum Talimatları

1. Ortam Kurulumu
```bash
git clone https://github.com/feyzakir/FDeep.AI.git
cd FDeep.AI
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

------------

Takım Adı: FDeep.AI
Teknofest 2025 Türkçe Doğal Dil İşleme Yarışması Serbest Kategori için oluşturulmuştur.

------------

Takım Üyeleri
Feyza Kıranlıoğlu GitHub: https://github.com/feyzakir
Derin Çıvgın GitHub: https://github.com/Derincvgn

------------

2. PDF Veri Setini Ekleme
Proje, `dijitalpazarlama_reklamkaynakları` adlı klasör altında PDF belgesiyle birlikte gelir.
Bu klasör, `main.py` çalıştırıldığında otomatik olarak yüklenir. Ekstra bir bağlantıya ihtiyaç yoktur.

 ------------

3. Uygulamayı Başlatma
```bash
streamlit run main.py


