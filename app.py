
# Accessibility: larger, easy-to-read text throughout the dashboard
st.markdown("""
<style>
/* Main text */
.stApp,
.stApp p,
.stApp li,
.stApp label,
.stApp [data-testid="stCaptionContainer"] {
    font-size: 19px !important;
    line-height: 1.65 !important;
}

/* Main headings */
.stApp h1 {
    font-size: 38px !important;
    line-height: 1.3 !important;
}
.stApp h2 {
    font-size: 30px !important;
}
.stApp h3 {
    font-size: 25px !important;
}

/* Sidebar: make all text clearly readable */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    font-size: 19px !important;
    line-height: 1.6 !important;
}

/* Metric cards and values */
[data-testid="stMetricLabel"] {
    font-size: 18px !important;
}
[data-testid="stMetricValue"] {
    font-size: 32px !important;
}
[data-testid="stMetricDelta"] {
    font-size: 17px !important;
}

/* Buttons */
.stButton button,
.stDownloadButton button {
    font-size: 19px !important;
    min-height: 48px !important;
    padding: 10px 16px !important;
}

/* Input text and selected options */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="select"] {
    font-size: 18px !important;
}

/* Make chart labels larger where SVG text styling applies */
.stApp svg text {
    font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)
