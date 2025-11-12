import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# ======================
# CONFIGURATION STREAMLIT
# ======================
st.set_page_config(page_title="Analyse des sentiments - Twitter Wave", layout="wide")

st.title("🌊 Analyse des Sentiments des Réponses Twitter - Wave CI")
st.markdown("Cette application scrape les tweets de **Wave CI**, analyse les réponses et en déduit les **sentiments et recommandations**.")

# ======================
# SECTION 1 : UPLOAD / SCRAPING
# ======================
st.header("1️⃣ Chargement ou Scraping des Données")

tab1, tab2 = st.tabs(["📂 Importer fichiers CSV", "🕷️ Scraper Twitter (manuel)"])

with tab1:
    tweets_file = st.file_uploader("Uploader le fichier de tweets (wave_civ_tweets.csv)", type=["csv"])
    replies_file = st.file_uploader("Uploader le fichier de réponses (wave_civ_reponses.csv)", type=["csv"])

    if tweets_file and replies_file:
        tweets = pd.read_csv(tweets_file)
        reponses = pd.read_csv(replies_file)
        st.success(f"{len(tweets)} tweets et {len(reponses)} réponses chargés avec succès.")
    else:
        st.warning("Veuillez importer les deux fichiers CSV.")

with tab2:
    st.info("⚠️ Le scraping via Selenium nécessite une interaction manuelle et ne peut pas être exécuté directement dans Streamlit Cloud.")
    st.code("Exécutez le script scraping séparément avant d'importer les fichiers ici.", language="python")

# ======================
# SECTION 2 : ANALYSE DE SENTIMENT
# ======================
if 'reponses' in locals():
    st.header("2️⃣ Analyse de Sentiment")

    nltk.download('stopwords')
    nltk.download('wordnet')

    stop_words = set(stopwords.words('french'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = str(text)
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^A-Za-zÀ-ÿ\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_text(text):
        tokens = text.lower().split()
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
        return " ".join(tokens)

    def get_sentiment(text):
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "positif"
        elif polarity < -0.1:
            return "negatif"
        else:
            return "neutre"

    with st.spinner("🧹 Nettoyage et analyse en cours..."):
        reponses["contenu_nettoye"] = reponses["contenu"].apply(clean_text)
        reponses["contenu_pretraite"] = reponses["contenu_nettoye"].apply(preprocess_text)
        reponses["sentiment"] = reponses["contenu_pretraite"].apply(get_sentiment)
        reponses["polarite"] = reponses["contenu_pretraite"].apply(lambda x: TextBlob(x).sentiment.polarity)

    st.success("Analyse terminée ! ✅")

    # ======================
    # SECTION 3 : VISUALISATION
    # ======================
    st.header("3️⃣ Visualisation des Résultats")

    sentiment_counts = reponses["sentiment"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Répartition des Sentiments")
        fig, ax = plt.subplots(figsize=(5, 4))
        sentiment_counts.plot(kind="bar", color=["green", "red", "gray"], edgecolor="black", ax=ax)
        plt.title("Répartition des Sentiments")
        st.pyplot(fig)

    with col2:
        st.subheader("📋 Statistiques")
        st.write(f"**Total réponses :** {len(reponses)}")
        for s, c in sentiment_counts.items():
            st.write(f"**{s.capitalize()} :** {c} ({c / len(reponses) * 100:.1f}%)")

    # ======================
    # SECTION 4 : TOP RÉPONSES
    # ======================
    st.header("4️⃣ Exemples de Réponses")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("😊 Positifs")
        top_pos = reponses[reponses["sentiment"] == "positif"].nlargest(10, "polarite")
        st.dataframe(top_pos[["auteur", "contenu", "polarite"]])
    with col2:
        st.subheader("😡 Négatifs")
        top_neg = reponses[reponses["sentiment"] == "negatif"].nsmallest(10, "polarite")
        st.dataframe(top_neg[["auteur", "contenu", "polarite"]])

    # ======================
    # SECTION 5 : NUAGES DE MOTS
    # ======================
    st.header("5️⃣ Nuages de mots")

    def make_wordcloud(texts, color="Greens"):
        txt = " ".join(texts)
        txt = re.sub(r"http\S+|www\S+", "", txt)
        txt = re.sub(r"@\w+", "", txt)
        txt = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ\s\-']", " ", txt.lower())
        wc = WordCloud(width=600, height=400, background_color="white", colormap=color, max_words=100).generate(txt)
        return wc

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Nuage de mots POSITIFS")
        wc_pos = make_wordcloud(reponses[reponses["sentiment"] == "positif"]["contenu_pretraite"], "Greens")
        st.image(wc_pos.to_array(), use_container_width=True)
    with col2:
        st.subheader("Nuage de mots NÉGATIFS")
        wc_neg = make_wordcloud(reponses[reponses["sentiment"] == "negatif"]["contenu_pretraite"], "Reds")
        st.image(wc_neg.to_array(), use_container_width=True)

    # ======================
    # SECTION 6 : RECOMMANDATIONS
    # ======================
    st.header("6️⃣ Recommandations Automatiques")

    negative_comments = reponses[reponses['sentiment'] == 'negatif']
    all_words = ' '.join(negative_comments['contenu_pretraite']).split()
    word_freq = Counter(all_words)
    most_common_words = [w for w, _ in word_freq.most_common(15)]

    recommandations = []

    if any(w in most_common_words for w in ["carte", "visa"]):
        recommandations.append("💳 Améliorer la disponibilité et la compatibilité des cartes Wave Visa.")
    if any(w in most_common_words for w in ["application", "bug", "connexion", "erreur"]):
        recommandations.append("📱 Optimiser la stabilité et corriger les bugs de l'application mobile.")
    if any(w in most_common_words for w in ["service", "client", "support", "assistance"]):
        recommandations.append("🤝 Renforcer le service client et la réactivité du support.")
    if any(w in most_common_words for w in ["retrait", "argent", "transfert", "paiement"]):
        recommandations.append("💰 Améliorer la rapidité et la fiabilité des transactions financières.")
    if any(w in most_common_words for w in ["frais", "tarif", "prix"]):
        recommandations.append("💸 Revoir la politique tarifaire, surtout pour les petites transactions.")

    if recommandations:
        for rec in recommandations:
            st.write(f"- {rec}")
    else:
        st.info("Aucune recommandation particulière : les commentaires négatifs ne révèlent pas de tendance forte.")
