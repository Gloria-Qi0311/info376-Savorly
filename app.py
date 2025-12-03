import streamlit as st
from normalize import hybrid_search
from recommendation import Recommender

st.set_page_config(page_title="Savorly Search", page_icon="🍳")
if 'recommender' not in st.session_state:
    st.session_state_recommender = Recommender()
    
st.title("Savorly - Recipe Hybrid Recommender System")
query = st.text_input("what are you craving today?", placeholder="e.g hot chicken snack")

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

with st.sidebar:
    st.header("Recommendations")
    st.write("test")