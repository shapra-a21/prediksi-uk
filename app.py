import streamlit as st
import joblib
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("model/model.h5")
tokenizer = joblib.load("model/tokenizer.pkl")
mlb = joblib.load("model/mlb.pkl")
label_kategori = mlb.classes_

MAX_LEN = 100

def preprocess_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split() 
    return " ".join(tokens)

keywords = {
    "UK_victim": [
        "korban", "pikmi", "salah cwe", "salah cewe", "nyalahin",
        "baju", "pakaian", "terbuka", "emang mau", "kenapa gak nolak",
        "ngapain keluar malem", "harusnya diem", "salah sendiri",
        "wajar digituin", "pantes dilecehkan", "pelecehan", "badan"
    ],
    "UK_patriarki": [
        "patriarki", "feminis", "cwe masak", "cewe masak" "cewek di dapur", "cewe dapur",
        "cewek gak cocok kerja", "wanita harus nurut", "istri harus taat",
        "perempuan ga usah sekolah", "cewek tempatnya di rumah",
        "bisa masak", "harus bisa masak", "cewek harus masak", "cewe kerja", "setara", "kesetaraan gender" 
    ],
    "UK_misogyny": [
        "goblok", "tolol", "matre", "murahan", "kuliah", "dapur", "betina",
        "nyusahin", "perawan tua", "gatel", "ga laku", "siapa suruh", "cewe hamil",
        "cewek bodoh", "ngapain jadi cewek", "bikin ribet", "cewek brengsek",
        "brengsek lu cewek", "susah ngertiin cewek", "cewek gak berguna", "manja banget"
    ],
    "UK_mockery": [
        "pikmi", "sensi mba", "udah jelek", "cewe gatau diri", "cwe gatau diri",
        "hidup susah", "takut hidup susah", "sok kuat", "bucin banget",
        "cuma modal cantik", "nangis mulu", "mau enaknya doang", "cewek manja",
        "gak bisa mikir", "sok feminis", "pura-pura kuat", "cewek keras kepala",
        "gatau diri", "sensi amat", "sensi", "cewe kesusahan", "cewe susah", "cantik najis"
    ],
    "Non_UK": [
        "baik hati", "baik cantik", "cantik nurut", "imut cantik", "masya allah",
        "tabarakallah", "anak baik", "anak cantik", "baik kamu",
        "kamu cantik", "baik banget", "cantik banget", "cantik luar dalam", "baik luar dalam",
        "wanita baik", "perempuan cerdas", "perempuan sukses", "wanita kuat", "cewek rajin",
        "perempuan hebat", "support wanita", "cewek baik",
        "cewek tangguh", "wanita tangguh", "doa terbaik", "semangat terus", "selalu kuat",
        "hebat banget", "salut sama cewek", "terima kasih", "kamu hebat", "tetap semangat",
        "wanita mandiri", "wanita sabar", "cewek mandiri",
        "cewek keren", "wanita keren"
    ]
}

# ========== Streamlit App ==========
st.set_page_config(page_title="Prediksi Ujaran Kebencian", layout="wide")

menu = st.sidebar.selectbox("Pilih Menu", ["Home", "Cek Prediksi"])

if menu == "Home":
    st.title("Prediksi Ujaran Kebencian terhadap Perempuan")
    st.markdown("""
    Website ini bertujuan untuk mendeteksi ujaran kebencian terhadap wanita berdasarkan komentar sosial media Twitter
    menggunakan implementasi model deep learning **Bidirectional Long-Short Term Memory (BiLSTM)**.

    - Dapat melakukan hasil prediksi single label dan multilabel.
    - Mendeteksi komentar sesuai dengan label kategori uk_victim, uk_misogyny, uk_mockery, uk_patriarki, dan non_uk, dengan keterangan sebagai berikut :

        - **uk_victim** : ujaran kebencian menyalahkan wanita.
        - **uk_misogyny** : ujaran kebencian perilaku patriarki terhadap wanita.
        - **uk_mockery** : ujaran kebencian langsung merendahkan wanita dengan pola pikir strereotip gender dan diskriminasi terhadap wanita.
        - **uk_patriarki** : ujaran kebencian menyindir dan mengejek wanita. 
        - **non_uk** : tidak termasuk ujaran kebencian terhadap wanita.

    - Prediksi dilakukan dengan menginput komentar berbasis teks atau dengan memilih dari contoh komentar yang tersedia.
    """)

    st.image("image/uk_twt.jpg", caption="Klasifikasi Ujaran Kebencian terhadap Wanita dengan BiLSTM", use_container_width=True)


elif menu == "Cek Prediksi":
    st.markdown("<h1 style='text-align: center;'> Prediksi Ujaran Kebencian</h1>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_input = st.text_input(" Masukkan komentar:", placeholder="Contoh: perempuan gausah kuliah tinggi...")

    with col2:
        examples = [
            "maunya cari cewek pinter masak, gausah sekolah kuliah tinggi. harusnya ngaca",
            "cewek tuh emang dasarnya manja banget, makanya ribet kalau kerja bareng",
            "wanita itu kuat dan hebat, jangan pernah diremehkan",
            "perempuan gausah kuliah tinggi, percuma ujungnya di dapur juga"
        ]  
        # Tetap seperti punyamu
        selected_example = st.selectbox(" Pilih contoh komentar:", options=[""] + examples)

    input_text = selected_example if selected_example else user_input

    if st.button("Prediksi"):
        if not input_text.strip():
            warning("Harap masukkan komentar terlebih dahulu.")
        else:
        # 1. Preprocessing
            cleaned = preprocess_and_tokenize(input_text)

        # 2. Keyword Matching (lowercase)
        input_lower = input_text.lower()
        keyword_labels = [
            label for label, kwlist in keywords.items()
            if any(kw in input_lower for kw in kwlist)
        ]

        # 3. Tokenize & Pad
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, padding='post', maxlen=MAX_LEN)

        # 4. Predict
        probs = model.predict(padded)[0]

        # 5. Adjust Probabilities (Optional)
        adjusted_probs = []
        for label, score in zip(label_kategori, probs):
            if label == "Non_UK":
                adjusted_probs.append(score * 0.7)
            else:
                adjusted_probs.append(score)

        # 6. Logic Final Label
        if keyword_labels:
            final_labels = keyword_labels
            score_map = dict(zip(label_kategori, adjusted_probs))
            final_scores = {lab: score_map.get(lab, 0.0) for lab in final_labels}
        else:
            threshold = 0.1
            model_labels = [
                lab for lab, sc in zip(label_kategori, adjusted_probs) if sc >= threshold
            ]
            if not model_labels:
                pairs_sorted = sorted(zip(label_kategori, adjusted_probs), key=lambda x: x[1], reverse=True)
                for lab, _ in pairs_sorted:
                    if lab != "Non_UK":
                        model_labels = [lab]
                        break
                else:
                    model_labels = [pairs_sorted[0][0]]
            final_labels = model_labels
            final_scores = {lab: dict(zip(label_kategori, adjusted_probs)).get(lab, 0.0) for lab in final_labels}

        # 7. Output
        st.markdown("### Hasil Prediksi")
        with st.container():
            st.markdown(
                f"""
                <div style="background-color:#f0f8ff; padding:20px; border-radius:10px;">
                    <b>Teks Input:</b><br>{input_text}<br><br>
                    <b>Deteksi Label:</b><br>{', '.join(keyword_labels) if keyword_labels else '— Tidak ditemukan —'}<br><br>
                    <b>Label Prediksi Akhir:</b><br>{', '.join(final_labels)}
                </div>
                """,
                unsafe_allow_html=True,
            )