import streamlit as st
import pandas as pd
import PyPDF2
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from lime.lime_text import LimeTextExplainer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import io
import spacy

st.set_page_config(
    page_title="Agent IA de preselection de cvs", 
    page_icon="🧑‍💻",
    layout="centered"
) 

@st.cache_resource
def load_model():
    return spacy.load("processed_resumes")

st.header("✨condidats/RH✨")
st.title("Agent IA de préselection de CVs")


nlp = spacy.load("en_core_web_lg")

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

def extract_name_and_contact(text):
    name = None
    email = None
    phone = None

    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    email_matches = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_matches:
        email = email_matches[0]

    phone_matches = re.findall(r'((?:\+216|00216)?\s?0?[2-9]\d{1}[-.\s]?\d{3}[-.\s]?\d{3,4})', text)
    if phone_matches:
        phone = phone_matches[0]

    return name, email, phone


dataset = load_dataset("cnamuangtoun/resume-job-description-fit")
df = pd.DataFrame(dataset["train"])
df["text"] = df["resume_text"] + " " + df["job_description_text"]
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vectorized, y_train)
class_names = model.classes_

def predict_proba(text_instance):
    if isinstance(text_instance, str):
        text_instance = [text_instance]
    elif not isinstance(text_instance, list):
        text_instance = list(text_instance)

    vectorized_text = vectorizer.transform(text_instance)
    return model.predict_proba(vectorized_text)

def get_lime_explanation(text_to_explain):
    explainer = LimeTextExplainer(class_names=class_names)
    explanation = explainer.explain_instance(
        text_to_explain,
        predict_proba,
        num_features=10
    )
    return explanation


with st.sidebar:
    st.image('logo.png', use_container_width=True, output_format='PNG')
    st.header("Description du poste")
    job_description = st.text_area(
    "Entrez la description du poste:",
    height=200,
    placeholder="Entrez la description du poste ici...")
    st.markdown("---")
    st.markdown("### A propos de cette application")
    st.markdown("""Cette application vous permet de:
1. Télécharger plusieurs CV au format PDF
2. Extraire automatiquement les noms et contacts des candidats
3. Évaluer la correspondance avec une description de poste
4. Sélectionner les meilleurs candidats""")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Télécharger les CVs")
    uploaded_files = st.file_uploader(
        "Choisissez les fichiers PDF",
        type="pdf",
        accept_multiple_files=True,
        help="Téléchargez plusieurs fichiers CV au format PDF"
    )
with col2:
    st.header("Résultats de l'analyse")
    
    if uploaded_files and job_description:
        candidates = []
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
                name, email, phone = extract_name_and_contact(text)
                combined_text = text + " " + job_description
                proba = predict_proba(combined_text)
            
                fit_score = proba[0][1] if len(proba[0]) > 1 else proba[0][0]

                text_vec = vectorizer.transform([text])
                job_vec = vectorizer.transform([job_description])
                sim_score = cosine_similarity(text_vec, job_vec)[0][0]

                candidates.append({
                    "Nom": name or "Unknown",
                    "Email": email or "Not found",
                    "Téléphone": phone or "Not found",
                    "Fit Score": fit_score,
                    "Similarity Score": sim_score,
                    "Combined Text": combined_text
                })
        if candidates:
            df = pd.DataFrame(candidates)
            df = df.sort_values(by="Fit Score", ascending=False)
            st.dataframe(df, use_container_width=True)

            st.subheader("Classement des candidats")
            for idx, candidate in df.iterrows():
                with st.expander(f"🧑‍💼 {candidate['Nom']} - Score: {candidate['Fit Score']:.3f} - Similarity: {candidate['Similarity Score']:.3f}"):
                    col_info, col_contact = st.columns([2, 1])
                    
                    with col_info:
                        st.write(f"**Fit Score:** {candidate['Fit Score']:.3f}")
                        if st.button(f"Voir l'explication pour {candidate['Nom']}", key=f"explain_{idx}"):
                            with st.spinner("Génération de l'explication..."):
                                try:
                                    explanation = get_lime_explanation(candidate['Combined Text'])
                                    
                                    st.subheader("LIME Explanation")
                                    st.write("Caractéristiques qui ont contribué à la prédiction:")
                                    explanation_text = explanation.as_list()
                                    for feature, weight in explanation_text:
                                        color = "green" if weight > 0 else "red"
                                        st.markdown(f"<span style='color: {color}'>{feature}: {weight:.3f}</span>", unsafe_allow_html=True)
                                        explanation_html = explanation.as_html()
                                    st.components.v1.html(explanation_html, height=400, scrolling=True)
                                    
                                except Exception as e:
                                    st.error(f"Erreur lors de la génération de l'explication: {str(e)}")

                    with col_contact:
                        st.write("**Informations de contact:**")
                        st.write(f"📧 {candidate['Email']}")
                        st.write(f"📞 {candidate['Téléphone']}")
            st.subheader("Résumé")
            st.write(f"📑 Total des candidats traités: {len(df)}")
            st.write(f"✅ Score de correspondance moyen: {df['Fit Score'].mean():.3f}")
            st.write(f"🏆 Meilleur candidat: {df.iloc[0]['Nom']} - Score: {df.iloc[0]['Fit Score']:.3f} - Similarity: {df.iloc[0]['Similarity Score']:.3f}")

    elif not uploaded_files:
        st.info("Veuillez télécharger des fichiers CV pour commencer l'analyse.")
    elif not job_description:
        st.info("Veuillez entrer une description de poste dans la barre latérale.")


