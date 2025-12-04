import streamlit as st
from normalize import hybrid_search
from recommendation import Recommender
import pandas as pd

def format_recipe(text):
    if isinstance(text, str):
        parts = [p.strip() for p in text.split(".") if p.strip()]
        if parts:
            return parts[0], parts[1:]
    return "recipe", []

st.set_page_config(page_title="Savorly Search", page_icon="🍳")
if 'recommender' not in st.session_state:
    st.session_state.recommender = Recommender()

st.title("Savorly - Recipe Hybrid Recommender System")
query = st.text_input("what are you craving today?", placeholder="e.g hot chicken snack")

@st.cache_data
def load_recipes():
    return pd.read_csv("output.csv")
recipes_df = load_recipes()

slider_val = st.slider(
    label="Select whether you want your recommendation to be more precise or semantic based on your search",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.10
)
st.write(slider_val)
if st.button("Search", type="primary"):
    if query:
        st.session_state.recommender.log_query(query)
        results = hybrid_search(query, b=slider_val)
        st.subheader("Results")
        if not results.empty:
            for index, row in results.head(10).iterrows():
                doc_idx = int(row['docno'])
                try:
                    full_text = recipes_df.iloc[doc_idx]['text']
                    title, body = format_recipe(full_text)
                except:
                    title, body = "recipe error", ["Content not available"]
                with st.expander(f"{title} (Score: {row['hybrid_score']:.2f})"):
                    for part in body:
                        st.write(part)
        else:
            st.write("No results found.")

with st.sidebar:
    st.header("Recommendations")
    if st.session_state.recommender.history:
        if st.button("Retrieve Recommendations"):
            st.session_state.last_recommend = st.session_state.recommender.generate_recommendations()