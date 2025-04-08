import streamlit as st
import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer

st.set_page_config(page_title="Article Tagger")

CATEGORIES = [
    "Computer Science",
    "Economics",
    "Statistics",
    "Math",
    "Electrical Engineering ans Systems Science",
    "Physics",
    "Quantitative Biology",
    "Quantitative Finance",
]

@st.cache_resource(show_spinner=False)
def load_model():
    checkpoint = "./tagger_model"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = RobertaForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs)["logits"].squeeze()
    result = []
    for ind, logit in enumerate(logits):
        if logit > 0:
            result.append(CATEGORIES[ind])
    return result

def reset_fields():
    st.session_state.title = ""
    st.session_state.abstract = ""

st.title("🧠 Article Tagger")

title_text = st.text_input("Title", placeholder="Enter article title here", key="title")
abstract_text = st.text_area("Abstract (optional)", placeholder="Enter abstract here", height=200, key="abstract")

col1, col2 = st.columns(2)
with col1:
    analyze = st.button("Analyze", use_container_width=True)
with col2:
    reset = st.button("Reset", use_container_width=True, on_click=reset_fields)

if analyze:
    if not title_text.strip():
        st.warning("Title is required!")
    else:
        full_text = title_text + " " + abstract_text
        with st.spinner("Analyzing..."):
            tags = predict(full_text)
        if tags:
            st.success("Tags detected:")
            for tag in tags:
                st.markdown(f"- ✅ **{tag}**")
        else:
            st.info("No tags detected.")
