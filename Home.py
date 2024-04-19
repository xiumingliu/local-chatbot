import streamlit as st

st.set_page_config(
    page_title="Welcome",
    page_icon='💬',
    layout='wide'
)

st.header(":star: Welcome to AI-augmented human intelligence! :star:")
st.text("")
st.text("")
st.text("")

col1, col2, col3 = st.columns([0.4, 0.1, 0.5])
with col1:
    st.image('./assets/ai_assistant_2.png')
with col3:
    st.markdown("### Who we are?")
    st.markdown("We are a team of AI consultant powered by large language models and many different tools.")
    st.markdown("### What can we support?")
    st.markdown("Here are some examples tasks which we can support:")
    st.markdown("- Search for relevant information")
    st.markdown("- Q&A with documents of hunderd pages")
    st.markdown("- Discover insights from your data")