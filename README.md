📚 **BERT-Based Book Recommendation System**
Welcome to the BERT-powered Book Recommender, a content-based recommendation system that suggests books similar to a given title using deep semantic understanding (via BERT embeddings). Built with torch, sentence transformers and rapidfuzz and deployed on Streamlit.

 **Features:**
 **Fuzzy Matching:** Intelligent title matching using partial input (e.g., "Harry Potter" finds "Harry Potter: The Goblet of Fire").

 **BERT Embeddings:** Uses sentence-transformers to understand semantic similarity in book descriptions.

 **Content-Based Recommendations:** No user data needed. Purely based on book descriptions and titles.

 **Customizable Filters:**

      Minimum Rating
      
      Language Code (e.g., "en", "fr")
      
      Publication Year
      
      Author

 **Streamlit UI:** Simple, interactive, and clean interface for quick exploration.

  **How It Works**
**Data Preprocessing:**

Each book's title and description are combined into a text block.

BERT (all-MiniLM-L6-v2) encodes this into a dense vector representation.

**User Input:**

The user provides a partial or full book title.

Optional filters (rating, year, author, language) can be applied.

**Fuzzy Title Matching:**

Uses rapidfuzz to find the best matching title in the dataset if no exact match is found.

**Semantic Similarity Search:**

Computes cosine similarity between the selected book’s embedding and all others.

Returns the top N similar books.

 **Dataset Format**
Expected columns in books.csv:

title: Title of the book

description: Summary or plot

average_rating: Average rating

original_publication_year: Year of publication

language_code: Language (e.g., en, fr)

author: Author name

You can easily adapt the script to work with other datasets in similar formats.

 **Credits**
Sentence-Transformers

RapidFuzz

Kaggle

Streamlit
Book metadata (source: Kaggle)
