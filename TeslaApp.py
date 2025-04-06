import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# read in the file 
tesla = pd.read_csv('tesla_used_car_sold-2022-08.csv')
print(tesla) 

#basic cleaning 

tesla.dropna(subset=['model','color','sold_price'],inplace=True)


tesla['sold_price'] = pd.to_numeric(tesla['sold_price'],errors='coerce')
tesla['miles'] = pd.to_numeric(tesla['miles'],errors='coerce')

if 'trim' in tesla.columns:
    trims = tesla['trim'].dropna().unique()
    selected_trims = st.sidebar.multiselect("Select Trim(s):", trims,default=list(trims))
    tesla = tesla[tesla['trim'].isin(selected_trims)]

st.title("Tesla Sales Explorer Dashboard")

# Sidebar Filters 
model_options = st.sidebar.multiselect("Select Tesla Model(s):", tesla['model'].unique(), default=tesla['model'].unique())
print(model_options)
color_options = st.sidebar.multiselect("Select Tesla Color(s):", tesla['color'].unique(), default= tesla['color'].unique() )
print(color_options)
price_range = st.sidebar.slider(
    "Select Sold Price Range:",
    int(tesla['sold_price'].min()),
    int(tesla['sold_price'].max()),
    (30000, 90000)
)
print(price_range)

if 'year' in tesla.columns:
    tesla['year'] = pd.to_numeric(tesla['year'],errors='coerce')
    min_year = int(tesla['year'].min())
    max_year = int(tesla['year'].max())
    selected_year_range = st.sidebar.slider(
        "Select Year Range:",
        min_year,
        max_year,
        (min_year, max_year)
    )
    tesla = tesla[tesla['year'].between(*selected_year_range)]

#Apply filters 
filtered_tesla = tesla[
    (tesla['model'].isin(model_options)) & 
    (tesla['color'].isin(color_options)) &
    (tesla['sold_price'].between(price_range[0],price_range[1]))
]
print(filtered_tesla) 

#Summary 
st.subheader("Summary Satistics")
st.write(filtered_tesla.describe())

#most popular models
st.subheader("Top Tesla Models in Selection")
st.bar_chart(filtered_tesla['model'].value_counts())

# most popular  colors 
st.subheader("Top Tesla Colors")
st.bar_chart(filtered_tesla['color'].value_counts())

# Sold Price Distribution
st.subheader("Sold Price Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_tesla['sold_price'], bins=20, kde=True, ax=ax)
ax.set_title("Sold Price ($)")
st.pyplot(fig)

st.subheader("📉 Sold Price vs Mileage")
fig2, ax2 = plt.subplots()
sns.scatterplot(data=filtered_tesla, x='miles', y='sold_price', hue='model', ax=ax2)
ax2.set_xlabel('Miles')
ax2.set_ylabel("Sold Price ($)")
st.pyplot(fig2)

st.subheader("Top 3 Best Value Teslas")

value_df = filtered_tesla.dropna(subset=['sold_price','miles'])

value_df = value_df.copy()
value_df['value_score'] =(1/value_df['sold_price']) + (1/value_df['miles']+1)

n_top = st.sidebar.slider("How many top value Teslas to show?", 1,10,3)

top_value_cars = value_df.sort_values('value_score', ascending=False).head(n_top)

st.write("These teslas have the best combo of low miles and price")

st.write(top_value_cars[['model','color','miles','sold_price']])

st.subheader("Download Filtered Data")

st.download_button(
    label="Download Filtered Data as CSV",
    data=filtered_tesla.to_csv(index=False),
    file_name='filtered_tesla_data.csv',
    mime='text/csv'
)

st.subheader("Miles vs Sold Price (with Value Highlights)")

fig , ax = plt.subplots()

sns.scatterplot(data= filtered_tesla , x ='miles', y = 'sold_price', hue = 'model', alpha = 0.6, ax=ax)

sns.scatterplot(data= top_value_cars, x='miles', y= 'sold_price', color='red',s =150,label ='Best Value', ax=ax)

ax.set_title("Tesla Price vs Miles")

ax.set_xlabel("Miles")
ax.set_ylabel("Sold Price ($)")
st.pyplot(fig)

st.subheader("Tesla Color Distrubution(Filtered Data)")

color_counts = filtered_tesla['color'].value_counts()

if not color_counts.empty:
    fig,ax =plt.subplots()
    ax.pie(
        color_counts,
        labels=color_counts.index,
        autopct='%1.1f%%',
        startangle=140
    )

    ax.axis('equal')
    st.pyplot(fig)
else:
    st.write("No data available for the selected filters.")

if 'year' in filtered_tesla.columns:
    st.subheader("Average Sold Price by Year")
    avg_price_by_year = (
        filtered_tesla.groupby('year')['sold_price'].mean().sort_index()
        )
    fig, ax = plt.subplots()
    avg_price_by_year.plot(kind ='line', marker= 'o',ax=ax)
    ax.set_ylabel("Average Sold Price($)")
    ax.set_xlabel("Year")
    ax.set_title("Tesla Price Trends by Year")
    st.pyplot(fig)

ml_df  = tesla.dropna(subset = ['sold_price','miles','model','year'] )

X  = ml_df[['miles','model','year']]
y = ml_df['sold_price']

preprocssor = ColumnTransformer(
    transformers=[('model_enc', OneHotEncoder(handle_unknown='ignore'),['model'])],
    remainder='passthrough'
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocssor),
    ('regressor', RandomForestRegressor(n_estimators  =100 , random_state = 42))
])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
pipeline.fit(X_train,y_train)

st.header("Predict Tesla Sold Price")

user_mileage = st.number_input("Enter miles:", min_value= 0, max_value=90000,value= 90000,step=1000)
user_year = st.selectbox("Select Year:",sorted(tesla['year'].dropna().unique(),reverse=True))
user_model = st.selectbox("Select Model:",tesla['model'].dropna().unique())

user_input = pd.DataFrame([{
    'miles': user_mileage,
    'year': user_year,
    'model': user_model
}])
predicted_price = pipeline.predict(user_input)[0]
st.success(f"Estimated Sold Price: ${predicted_price:,.2f}")

st.header("🤖 Ask the Tesla Bot")

user_question = st.text_input("Ask a question about Tesla cars:")

if user_question:
    response = "❓ Sorry, I didn’t understand that. Try asking about price, value, or popular models."

    question = user_question.lower()

    # Handle different types of basic queries
    if "cheapest" in question:
        cheapest = tesla.sort_values("sold_price").iloc[0]
        response = f"The cheapest Tesla sold was a {cheapest['model']} in {cheapest['color']} for ${cheapest['sold_price']:,}."

    elif "best value" in question:
        best = top_value_cars.iloc[0]
        response = (
            f"The best value Tesla is a {best['model']} ({best['color']}) "
            f"with {int(best['mileage'])} miles, sold for ${best['sold_price']:,}."
        )

    elif "average price" in question or "how much" in question:
        for model in tesla['model'].unique():
            if model.lower() in question:
                avg_price = tesla[tesla['model'] == model]['sold_price'].mean()
                response = f"The average price of a {model} is ${avg_price:,.2f}."
                break

    elif "popular" in question or "most owned" in question:
        most_common = tesla['model'].value_counts().idxmax()
        response = f"The most popular Tesla model in the dataset is the {most_common}."

    st.success(response)
