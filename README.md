Prototype complet d'IA pour booster la visibilité des créateurs de contenu sur les réseaux sociaux

import pandas as pd import numpy as np from sklearn.feature_extraction.text import TfidfVectorizer from sklearn.ensemble import RandomForestClassifier from sklearn.model_selection import train_test_split from sklearn.metrics import accuracy_score import re import string import openai import streamlit as st

Configuration API OpenAI (à sécuriser dans la version finale)

openai.api_key = "YOUR_OPENAI_API_KEY"

----------- Interface Streamlit -----------

st.set_page_config(page_title="AI Booster Réseaux Sociaux", layout="centered") st.title("📈 Analyseur de Posts pour Créateurs de Contenu") st.write("Améliore tes publications avec une IA experte en réseaux sociaux.")

Exemple de base de données (à remplacer par les données utilisateurs)

data = { 'text': [ "Nouvelle vidéo ! Donnez-moi votre avis ! #fun #video", "Regardez ma dernière astuce marketing !", "Juste une photo de mon chat 🐱", "Incroyable astuce pour gagner plus sur Instagram !", "C'est lundi... motivation 0 😩", "Partagez ce post si vous êtes d'accord !", ], 'likes': [120, 250, 30, 340, 20, 180], 'comments': [10, 50, 2, 80, 1, 35], 'shares': [5, 20, 0, 50, 0, 15], 'time_posted': ['10:00', '14:30', '21:00', '11:15', '08:00', '16:00'] }

df = pd.DataFrame(data) df['engagement'] = df['likes'] + df['comments']*2 + df['shares']*3 df['success'] = df['engagement'] > df['engagement'].median()

Prétraitement

def clean_text(text): text = text.lower() text = re.sub(f"[{string.punctuation}]", "", text) return text

df['clean_text'] = df['text'].apply(clean_text)

Vectorisation

vectorizer = TfidfVectorizer() X = vectorizer.fit_transform(df['clean_text']) y = df['success']

Entraînement du modèle

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) model = RandomForestClassifier(n_estimators=100) model.fit(X_train, y_train)

Interface utilisateur Streamlit

user_post = st.text_area("✍️ Colle ici ton post", "Écris quelque chose...") if st.button("Analyser mon post"): if user_post.strip() == "": st.warning("Merci d'écrire un post à analyser.") else: cleaned = clean_text(user_post) vec = vectorizer.transform([cleaned]) prediction = model.predict(vec)[0]

if prediction:
        st.success("✅ Ce post a de bonnes chances de bien performer !")
    else:
        st.error("🚫 Ce post risque de moins performer.")

        # Suggestion IA
        with st.spinner("Génération de suggestions IA..."):
            prompt = f"Voici une publication qui n'a pas bien fonctionné : '{user_post}'. Peux-tu suggérer une meilleure version qui inciterait plus d'engagement ?"
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "Tu es un expert en réseaux sociaux."},
                        {"role": "user", "content": prompt}
                    ]
                )
                suggestion = response.choices[0].message['content']
                st.markdown("### 💡 Suggestion IA :")
                st.write(suggestion)
            except Exception as e:
                st.error("Erreur lors de la génération de la suggestion IA.")
                st.exception(e)

Affichage des données d'entraînement (optionnel)

with st.expander("🔍 Voir les exemples de posts analysés"): st.dataframe(df[['text', 'engagement', 'success']])

