import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import torch
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sentence_transformers import SentenceTransformer, util
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import time
import sys
from bs4 import BeautifulSoup

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
model = SentenceTransformer('all-MiniLM-L6-v2')
geolocator = Nominatim(user_agent="food-recommender")

def normalize_food_name(text):
    text = text.lower()
    tokens = word_tokenize(text)
    cleaned = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words]
    return " ".join(cleaned)

def get_lat_lon(city_name):
    location = geolocator.geocode(city_name)
    if not location:
        raise ValueError("Invalid city name. Please enter a recognizable city.")
    return (location.latitude, location.longitude, city_name)

def get_swiggy_data(lat, lon):
    url = f"https://www.swiggy.com/dapi/restaurants/list/v5?lat={lat}&lng={lon}&is-seo-homepage-enabled=true&page_type=DESKTOP_WEB_LISTING"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers, timeout=10)
        data = res.json()
        cards = data.get('data', {}).get('cards', [])
        restaurants = []
        for card in cards:
            try:
                restaurants += card['card']['card']['gridElements']['infoWithStyle']['restaurants']
            except KeyError:
                continue
        return restaurants
    except Exception as e:
        print("⚠️ Error fetching data from Swiggy:", e)
        return []

def get_zomato_data(city_name):
    try:
        url = f"https://www.zomato.com/{city_name.lower().replace(' ', '-')}/restaurants"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        zomato_restaurants = []
        listings = soup.find_all('div', {'class': 'sc-clNaTc cCUpbB'})
        for item in listings:
            try:
                name = item.find('h4').text.strip()
                cuisines = item.find('p').text.strip().split(', ')
                zomato_restaurants.append({
                    'name': name,
                    'area': city_name,
                    'rating': 4.0,
                    'cuisines': cuisines,
                    'cost_for_two': 300,
                    'distance_km': 2.0,
                    'score': 0.0
                })
            except:
                continue
        return zomato_restaurants
    except Exception as e:
        print("⚠️ Error fetching data from Zomato:", e)
        return []

def filter_and_rank_restaurants(restaurants, user_food, budget, user_coords):
    user_embedding = model.encode(normalize_food_name(user_food), convert_to_tensor=True)
    results = []

    for res in restaurants:
        name = res.get('name', '')
        cuisines = res.get('cuisines', [])
        if not cuisines:
            continue

        rating = res.get('rating', 0.0)
        cost = res.get('cost_for_two', 300)
        area = res.get('area', '')
        distance_km = res.get('distance_km', 2.0)

        if budget != 1 and cost > budget + 50:
            continue

        food_lower = user_food.lower()
        cuisine_str = " ".join(cuisines).lower()
        name_lower = name.lower()

        if food_lower in cuisine_str or food_lower in name_lower:
            match_score = 1.0
        else:
            combined_text = cuisine_str + " " + name_lower
            food_embedding = model.encode(normalize_food_name(combined_text), convert_to_tensor=True)
            match_score = util.pytorch_cos_sim(user_embedding, food_embedding).item()

        if match_score < 0.15:
            continue

        final_score = (
            0.5 * match_score +
            0.25 * (rating / 5) +
            0.1 * (1 / (1 + distance_km)) +
            0.15 * (1 / (1 + cost))
        )

        res['score'] = round(final_score, 3)
        results.append(res)

    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:10]

# STREAMLIT APP STARTS HERE
st.set_page_config(page_title="🍽️ Budget Food Recommender", layout="wide", page_icon="🍴")

st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            color: #f63366;
            font-weight: bold;
            text-align: center;
        }
        .sub-text {
            font-size: 18px;
            color: #4f4f4f;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🍔 Budget Food Recommendation System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-text'>Find the best places to eat within your budget. Powered by Swiggy + Zomato + AI ❤️</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("📍 Enter Your Preferences")
    city = st.text_input("City", value="Delhi")
    food = st.text_input("Craving for (e.g., Momos, Pizza, Dosa)", value="Pizza")
    budget = st.slider("Budget in ₹", min_value=1, max_value=2000, value=300, step=20)
    search_btn = st.button("🔍 Find Restaurants")

if search_btn:
    try:
        with st.spinner("Fetching coordinates..."):
            lat, lon, city_name = get_lat_lon(city)

        with st.spinner("Scraping restaurant data from Swiggy & Zomato..."):
            swiggy_raw = get_swiggy_data(lat, lon)
            swiggy_restaurants = []
            for info in swiggy_raw:
                try:
                    swiggy_restaurants.append({
                        'name': info['info']['name'],
                        'area': info['info']['areaName'],
                        'rating': float(info['info'].get('avgRating', 0.0)),
                        'cuisines': info['info'].get('cuisines', []),
                        'cost_for_two': int(''.join(filter(str.isdigit, info['info'].get('costForTwo', '₹300')))) or 300,
                        'distance_km': 2.0,
                        'score': 0.0
                    })
                except:
                    continue

            zomato_restaurants = get_zomato_data(city_name)
            all_restaurants = swiggy_restaurants + zomato_restaurants

        st.success(f"✅ Found {len(all_restaurants)} total restaurants!")

        st.info("Ranking restaurants based on AI & your preferences...")
        top_matches = filter_and_rank_restaurants(all_restaurants, food, budget, (lat, lon, city_name))

        if not top_matches:
            st.warning("😕 No matching restaurants found. Try increasing your budget or changing food item.")
        else:
            st.markdown("### 🏆 Top Recommended Restaurants")
            df = pd.DataFrame(top_matches)
            st.dataframe(df[['name', 'area', 'rating', 'cuisines', 'cost_for_two', 'distance_km', 'score']])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg. Cost for Two", f"₹{int(df['cost_for_two'].mean())}")
            with col2:
                st.metric("Avg. Rating", round(df['rating'].mean(), 2))
            with col3:
                st.metric("Avg. Distance", f"{round(df['distance_km'].mean(), 2)} km")

            st.markdown("### 📊 Visual Insights")
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            df.plot(kind='barh', x='name', y='cost_for_two', ax=axs[0], color='skyblue', legend=False)
            axs[0].set_title('Cost for Two')
            df.plot(kind='barh', x='name', y='rating', ax=axs[1], color='lightgreen', legend=False)
            axs[1].set_title('Ratings')
            df.plot(kind='barh', x='name', y='distance_km', ax=axs[2], color='salmon', legend=False)
            axs[2].set_title('Distance from You')
            st.pyplot(fig)

    except Exception as e:
        st.error(f"🚫 Error: {e}")
