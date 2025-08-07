import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# Load everything once
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv("book.csv")
    df = df[['title', 'description', 'average_rating', 'authors', 'original_publication_year', 'language_code']].dropna()
    df = df[df['description'].str.strip() != '']
    df = df.drop_duplicates(subset='title').reset_index(drop=True)
    embeddings = torch.load("book_embeddings.pt", map_location=torch.device("cpu"))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return df, embeddings, model

df, embeddings, model = load_model_and_data()

st.title("📚 Book Recommendation")

# Input from user
title = st.text_input("Enter a book title", "The Great Gatsby")
top_n = st.slider("Number of recommendations", 1, 10, 5)
min_rating = st.slider("Minimum rating", 0.0, 5.0, 0.0)
lang = st.selectbox("Language code", ["", *df["language_code"].dropna().unique()])
year = st.number_input("Published after year", min_value=0, max_value=3000, value=0, step=1)
author = st.text_input("Author (optional)", "")

# Recommendation logic
def recommend_books():
    df_filtered = df.copy()
    if min_rating:
        df_filtered = df_filtered[df_filtered["average_rating"] >= min_rating]
    if lang:
        df_filtered = df_filtered[df_filtered["language_code"] == lang]
    if year:
        df_filtered = df_filtered[df_filtered["original_pubication_year"] >= year]
    if author:
        df_filtered = df_filtered[df_filtered["authors"].str.contains(author, case=False, na=False)]

    matches = df[df["title"].str.lower() == title.lower()]
    if matches.empty:
        st.warning("❌ Book title not found in dataset.")
        return []

    idx = matches.index[0]
    query_vec = embeddings[idx].unsqueeze(0)
    filtered_indices = df_filtered.index
    filtered_embeds = embeddings[filtered_indices]

    sim_scores = util.pytorch_cos_sim(query_vec, filtered_embeds)[0]
    top_scores = torch.topk(sim_scores, k=min(top_n, len(filtered_indices)))
    top_indices = filtered_indices[top_scores.indices.cpu().numpy()].tolist()

    return df_filtered.loc[top_indices]

# Display results
if st.button("Recommend"):
    results = recommend_books()
    if not results.empty:
        for _, row in results.iterrows():
            st.markdown(f"""
            ### 📖 {row['title']}
            - ✍️ Author: {row['authors']}
            - ⭐ Rating: {row['average_rating']}
            - 📅 Year: {int(row['original_publication_year'])}
            - 🌐 Language: {row['language_code']}
            - 📝 Description: {row['description'][:500]}...
            ---
            """)
    else:
        st.info("No recommendations found based on the selected filters.")
