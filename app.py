import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Personalized Product Recommender", page_icon="🛍️", layout="wide")

PRODUCTS = [
    {"id": 1, "name": "Wireless Bluetooth Headphones", "category": "Electronics", "price": 2499, "rating": 4.5, "description": "wireless bluetooth headphones noise cancellation deep bass music calls travel", "tags": "audio music tech travel"},
    {"id": 2, "name": "Smart Fitness Watch", "category": "Electronics", "price": 3299, "rating": 4.3, "description": "smart watch fitness tracker heart rate steps sleep sports health notifications", "tags": "fitness health wearable sports"},
    {"id": 3, "name": "Portable Bluetooth Speaker", "category": "Electronics", "price": 1599, "rating": 4.4, "description": "portable bluetooth speaker loud sound bass music outdoor rechargeable", "tags": "audio music travel"},
    {"id": 4, "name": "Mechanical Gaming Keyboard", "category": "Electronics", "price": 2899, "rating": 4.6, "description": "mechanical keyboard gaming RGB backlit keys computer gaming setup", "tags": "gaming computer accessories"},
    {"id": 5, "name": "Ergonomic Wireless Mouse", "category": "Electronics", "price": 899, "rating": 4.2, "description": "wireless mouse ergonomic computer office productivity laptop", "tags": "computer office accessories"},
    {"id": 6, "name": "Python Programming Guide", "category": "Books", "price": 599, "rating": 4.7, "description": "learn python programming coding beginner software development examples", "tags": "python coding programming education"},
    {"id": 7, "name": "Data Science Handbook", "category": "Books", "price": 799, "rating": 4.6, "description": "data science machine learning statistics pandas python artificial intelligence", "tags": "AI coding education data"},
    {"id": 8, "name": "Cybersecurity Fundamentals", "category": "Books", "price": 699, "rating": 4.8, "description": "cybersecurity ethical hacking network security linux penetration testing", "tags": "security hacking technology education"},
    {"id": 9, "name": "Minimalist Backpack", "category": "Fashion", "price": 1199, "rating": 4.1, "description": "minimalist backpack college laptop bag travel lightweight durable", "tags": "college travel bags"},
    {"id": 10, "name": "Running Sports Shoes", "category": "Fashion", "price": 2199, "rating": 4.4, "description": "running shoes sports sneakers comfortable lightweight workout fitness", "tags": "sports fitness clothing"},
    {"id": 11, "name": "Cotton Casual T-Shirt", "category": "Fashion", "price": 499, "rating": 4.0, "description": "cotton casual t-shirt comfortable everyday basic clothing fashion", "tags": "clothing casual"},
    {"id": 12, "name": "Desk Study Lamp", "category": "Home", "price": 999, "rating": 4.3, "description": "LED desk study lamp adjustable brightness reading study workspace", "tags": "home study office"},
    {"id": 13, "name": "Insulated Water Bottle", "category": "Home", "price": 399, "rating": 4.2, "description": "reusable insulated stainless steel water bottle gym school travel", "tags": "home fitness travel"},
    {"id": 14, "name": "Ceramic Coffee Mug Set", "category": "Home", "price": 649, "rating": 4.1, "description": "ceramic coffee tea mugs kitchen drinkware home gift", "tags": "home kitchen gifts"},
    {"id": 15, "name": "Yoga Mat", "category": "Sports", "price": 799, "rating": 4.5, "description": "non slip yoga mat exercise stretching meditation fitness workout", "tags": "fitness sports wellness"},
    {"id": 16, "name": "Resistance Bands Set", "category": "Sports", "price": 549, "rating": 4.3, "description": "resistance bands strength training home workout fitness exercise", "tags": "fitness sports workout"},
    {"id": 17, "name": "Web Development Course Book", "category": "Books", "price": 749, "rating": 4.5, "description": "HTML CSS JavaScript web development frontend programming beginner", "tags": "coding web education"},
    {"id": 18, "name": "Laptop Stand", "category": "Electronics", "price": 1399, "rating": 4.4, "description": "adjustable laptop stand ergonomic desk computer study productivity", "tags": "computer office study"},
]

df = pd.DataFrame(PRODUCTS)
df["content"] = df["category"] + " " + df["name"] + " " + df["description"] + " " + df["tags"]
vectorizer = TfidfVectorizer(stop_words="english")
matrix = vectorizer.fit_transform(df["content"])
similarities = cosine_similarity(matrix)

def recommend(product_id, category, interests, budget, count=5):
    profile = (category + " ") * 2 + " " + interests
    profile_vec = vectorizer.transform([profile])
    profile_scores = cosine_similarity(profile_vec, matrix).flatten()
    chosen = df[df["price"] <= budget].copy()
    if category != "All":
        chosen = chosen[chosen["category"] == category]
    chosen = chosen[chosen["id"] != product_id]
    if chosen.empty:
        return chosen
    chosen["score"] = chosen["id"].apply(lambda x: profile_scores[df.index[df["id"] == x][0]])
    chosen["score"] = chosen["score"] * 0.8 + (chosen["rating"] / 5.0) * 0.2
    return chosen.sort_values(["score", "rating"], ascending=False).head(count)

st.title("🛍️ Personalized Product Recommendation Engine")
st.write("Discover products selected for your interests, budget, and preferred category.")
st.caption("Demo project • Content-based filtering • TF-IDF + cosine similarity")

with st.sidebar:
    st.header("🎯 Personalize")
    interests = st.text_area("Your interests", "technology, coding, fitness", help="Example: music, travel, python, gaming")
    category = st.selectbox("Preferred category", ["All"] + sorted(df["category"].unique().tolist()))
    budget = st.slider("Maximum budget (₹)", 500, 5000, 3000, step=100)
    count = st.slider("Number of recommendations", 3, 8, 5)
    st.info("Recommendations use product descriptions and your selected preferences.")

tab1, tab2 = st.tabs(["✨ For You", "🔎 Explore Products"])
with tab1:
    st.subheader("Recommended for you")
    recs = recommend(None, category, interests, budget, count)
    if recs.empty:
        st.warning("No products match these filters. Increase your budget or choose All categories.")
    else:
        cols = st.columns(3)
        for i, (_, p) in enumerate(recs.iterrows()):
            with cols[i % 3]:
                st.markdown(f"### {p['name']}")
                st.markdown(f"**{p['category']}** · ⭐ {p['rating']}/5")
                st.markdown(f"## ₹{p['price']:,}")
                st.caption(p["description"].capitalize() + ".")
                st.progress(min(float(p["score"]), 1.0), text=f"Match score: {p['score']:.0%}")
                if st.button("Why this recommendation?", key=f"why_{p['id']}"):
                    st.info(f"This product matches your interests: {interests}. Its category and description are compared with your profile using TF-IDF and cosine similarity.")
                st.divider()
with tab2:
    st.subheader("Explore the catalog")
    query = st.text_input("Search products", placeholder="Try headphones, Python, fitness...")
    cat_filter = st.selectbox("Filter catalog by category", ["All"] + sorted(df["category"].unique().tolist()), key="catalog_category")
    view = df.copy()
    if cat_filter != "All":
        view = view[view["category"] == cat_filter]
    if query.strip():
        mask = (view["name"] + " " + view["description"] + " " + view["tags"]).str.contains(query, case=False, na=False)
        view = view[mask]
    st.write(f"{len(view)} product(s) found")
    st.dataframe(view[["name", "category", "price", "rating", "description"]].rename(columns={"name":"Product", "category":"Category", "price":"Price (₹)", "rating":"Rating", "description":"Description"}), use_container_width=True, hide_index=True)

with st.expander("📌 About this project"):
    st.markdown("""
    **How it works**
    1. Product names, categories, descriptions, and tags are combined into text.
    2. TF-IDF converts product text into numerical vectors.
    3. Cosine similarity compares your interests with each product.
    4. A small rating boost helps rank highly rated items.

    **Model type:** Content-based recommendation.  
    **Dataset:** Built-in sample catalog for demonstration; replace it with a real product dataset for deployment.
    """)

st.caption("Built with Python • Streamlit • Pandas • Scikit-learn")
