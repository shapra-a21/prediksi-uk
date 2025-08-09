import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

text = input("Masukkan teks: ")
X = vectorizer.transform([text])
y_pred = model.predict(X)

print("Hasil Prediksi:", y_pred)
