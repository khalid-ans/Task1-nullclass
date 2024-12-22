#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor


# In[3]:


import nltk


# In[4]:


import webbrowser
import os


# In[5]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[6]:


apps_df=pd.read_csv("Play Store Data.csv")


# In[7]:


user_df=pd.read_csv('User Reviews.csv')


# In[8]:


apps_df.head()


# In[9]:


#apps_df.drop_duplicate()


# In[10]:


apps_df.dropna(subset=['Rating'], inplace=True)


for column in apps_df.columns:
    if apps_df[column].mode().size > 0: 
        apps_df[column].fillna(apps_df[column].mode()[0], inplace=True)


apps_df.drop_duplicates(inplace=True)


apps_df = apps_df[apps_df["Rating"] <= 5]

user_df.dropna(subset=["Translated_Review"], inplace=True)


# In[11]:


apps_df.dtypes


# In[12]:


apps_df.head()


# In[ ]:





# In[13]:


#replacing + and object to int

apps_df['Installs'] = apps_df['Installs'].str.replace('+', '', regex=False).str.replace(',', '').astype(int)


# In[14]:


#replacing price into float 


# In[15]:


apps_df['Price'] = apps_df['Price'].str.replace('$', '', regex=False).astype(float)


# In[16]:


apps_df.info()


# In[ ]:





# In[17]:


user_df.head()


# In[18]:


apps_df.head(5)


# In[19]:


merge_df=pd.merge(apps_df,user_df,on='App',how='inner')


# In[20]:


merge_df.head()


# In[21]:


def convert(size):
    if 'M' in size:
        return float(size.replace('M',''))
    elif 'K' in size:
        return float(size.replace('K',''))/1024
    else:
        return np.nan
apps_df['Size']=apps_df['Size'].apply(convert)


# In[22]:


apps_df.head()


# In[23]:


apps_df['Log_installs']=np.log(apps_df['Installs'])


# In[24]:


apps_df.dtypes


# In[25]:


apps_df["Reviews"]=apps_df["Reviews"].astype(int)


# In[26]:


apps_df['Log_reviews']=np.log(apps_df['Reviews'])


# In[27]:


def rev(rating):
    if rating >=4:
        return 'Top rated'
    elif rating >=3:
        return 'Above average'
    elif rating >=2:
        return 'Average'
    else:
        return 'Below Average'


# In[28]:


apps_df['Rating']=apps_df['Rating'].apply(rev)


# In[29]:


apps_df.head()


# In[30]:


apps_df['Revenue']=apps_df['Installs']*apps_df['Price']


# In[31]:


inf=apps_df[1300:1310]


# In[32]:


inf


# In[33]:


apps_df.shape


# In[34]:


sia=SentimentIntensityAnalyzer()


# In[35]:


revi="I hope this product will be making you happy enough"


# In[36]:


sentiment_scroe=sia.polarity_scores(revi)


# In[37]:


sentiment_scroe


# In[38]:


revi2="I hope this product will be making you sad enough"


# In[39]:


sentiment_scroe=sia.polarity_scores(revi2)
sentiment_scroe


# In[40]:


#Senitment Score (compound): 
 # -1 :negative review
    #=1 : positive review


# In[41]:


user_df


# In[42]:


user_df['Sentiment Score']=user_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])


# In[43]:


user_df.head()


# In[44]:


apps_df['Year']=pd.to_datetime(apps_df['Last Updated'],errors='coerce')


# In[45]:


apps_df["years"]=apps_df['Year'].dt.year


# In[46]:


apps_df.head()


# In[47]:


html_files_path="./"
if not os.path.exists(html_files_path):
    os.makedirs(html_files_path)


# In[48]:


plot_containers=""


# In[49]:


def save_plot_as_html(fig,filename,insight):
    global plot_containers
    filepath=os.path.join(html_files_path,filename)
    html_content=pio.to_html(fig,full_html=False,include_plotlyjs='inline')
    
    #append the plot and its insight to plot_containers
    plot_containers+=f"""
    <div class="plot-container" id="{filename}"onclick="openPlot('{filename}')">
      <div class="insights">{html_content}</div>
      <div class="insights">{insight}</div>
    </div>
    """
    fig.write_html(filepath,full_html=False,include_plotlyjs='inline')


# In[50]:


plot_width=400
plot_height=300
plot_bg_color='black'
text_color='white'
title_font={'size':16}
axis_font={'size':12}


# In[51]:


category_counts=apps_df['Category'].value_counts().nlargest(10)
fig1=px.bar(
    x=category_counts.index,
    y=category_counts.values,
    labels={"x":"Category","y":"Count"},
    title="Top 10 Catgories",
    color=category_counts.index,
    color_discrete_sequence=px.colors.sequential.Plasma,
    height=300,
    width=400
    
)
fig1.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={"size":12}),
    margin=dict(t=30,l=10,r=10,b=10)
)

save_plot_as_html(fig1,"Category Graph 1.html","These are the top 10 catgories dominating on play Store")


# In[52]:


type_counts=apps_df['Type'].value_counts()
fig2=px.pie(
    values=type_counts.values,
    names=type_counts.index,
    title="App Type Distribution",
    color_discrete_sequence=px.colors.sequential.RdBu,
    height=300,
    width=400
    
)
fig2.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    margin=dict(t=30,l=10,r=10,b=10)
)

save_plot_as_html(fig2,"Type Graph 2.html","Whether the app is free or not to attract customers ")


# In[53]:


fig3=px.histogram(
    apps_df,
    x='Rating',
    nbins=20,
    title="Rating Distribution",
    color_discrete_sequence=['#636EFA'],
    height=300,
    width=400
    
)
fig3.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    margin=dict(t=30,l=10,r=10,b=10)
)

save_plot_as_html(fig3,"Rating Graph 2.html","Users Experience towards the application ")


# In[54]:


Sentiment_counts=user_df['Sentiment Score'].value_counts()
fig4=px.bar(
    x=Sentiment_counts.index,
    y=Sentiment_counts.values,
    labels={"x":"Sentiment_score","y":"Count"},
    title="Sentiment Score Distribution",
    color=Sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.RdPu,
    height=300,
    width=400
    
)
fig4.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={"size":12}),
    margin=dict(t=30,l=10,r=10,b=10)
)

save_plot_as_html(fig4,"Sentiment Score Graph 4.html","Sentiment in Reviews show a mix of Positive and Negative")


# In[55]:


installs_by_category=apps_df.groupby('Category')['Installs'].sum().nlargest(10)
fig5=px.bar(
    x=installs_by_category.index,
    y=installs_by_category.values,
    labels={"x":"Installs","y":"Category"},
    title="Installs by Category",
    color=installs_by_category.index,
    color_discrete_sequence=px.colors.sequential.Blues,
    height=300,
    width=400
    
)
fig4.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    font_color='white',
    title_font={'size':16},
    xaxis=dict(title_font={'size':12}),
    yaxis=dict(title_font={"size":12}),
    margin=dict(t=30,l=10,r=10,b=10)
)

save_plot_as_html(fig5,"Installs_by_category Graph 5.html","Which is most loveable category for developers")


# In[56]:


updates_per_year = apps_df['Last Updated'].value_counts().sort_index()

fig6 = px.line(
    x=updates_per_year.index,
    y=updates_per_year.values,
    labels={'x': 'Year', 'y': 'Number of Updates'},
    title='Number of Updates Over the Years',
    color_discrete_sequence=['#AB63FA'],
    width=plot_width,
    height=plot_height
)

fig6.update_layout(
    plot_bgcolor=plot_bg_color,
    paper_bgcolor=plot_bg_color,
    font_color=text_color,
    title_font=title_font,
    xaxis=dict(title_font=axis_font),
    yaxis=dict(title_font=axis_font),
    margin=dict(l=10, r=10, t=30, b=10)
)

save_plot_as_html(fig6, "Updates Graph 6.html","Shows the update date")


# In[57]:


revenue_by_category = apps_df.groupby('Category')['Revenue'].sum().nlargest(10)

fig7 = px.bar(
    x=revenue_by_category.index,
    y=revenue_by_category.values,
    labels={'x': 'Category', 'y': 'Revenue'},
    title='Revenue by Category',
    color=revenue_by_category.index,
    color_discrete_sequence=px.colors.sequential.Greens,
    width=400,
    height=300
)

fig7.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=dict(size=16),
    xaxis=dict(title_font=dict(size=12)),
    yaxis=dict(title_font=dict(size=12)),
    margin=dict(l=10, r=10, t=30, b=10)
)



save_plot_as_html(fig7, "Revenue Graph 7.html","Revenue generated by developer based on the category of Application")


# In[58]:


genre_counts = apps_df['Genres'].str.split(';', expand=True).stack().value_counts().nlargest(10)

fig8 = px.bar(
    x=genre_counts.index,
    y=genre_counts.values,
    labels={'x': 'Genre', 'y': 'Count'},
    title='Top Genres',
    color=genre_counts.index,
    color_discrete_sequence=px.colors.sequential.OrRd,
    width=400,
    height=300
)

fig8.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=dict(size=16),
    xaxis=dict(title_font=dict(size=12)),
    yaxis=dict(title_font=dict(size=12)),
    margin=dict(l=10, r=10, t=30, b=10)
)


save_plot_as_html(fig8, "Genre Graph 8.html","Showing the top genres")


# In[59]:


fig9 = px.scatter(
    apps_df,
    x='Last Updated',
    y='Rating',
    color='Type',
    title="Impact of Last Update on Rating",
    color_discrete_sequence=px.colors.qualitative.Vivid,
    width=400,
    height=300
)

fig9.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=dict(size=16),
    xaxis=dict(title_font=dict(size=12)),
    yaxis=dict(title_font=dict(size=12)),
    margin=dict(l=10, r=10, t=30, b=10)
)



save_plot_as_html(fig9, "Update Graph 9.html","Changes in Rating due to last update ")


# In[60]:


fig10 = px.box(
    apps_df,
    x='Type',
    y='Rating',
    color='Type',
    title='Rating for Paid vs Free Apps',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    width=400,
    height=300
)

fig10.update_layout(
    plot_bgcolor='black',
    paper_bgcolor='black',
    font_color='white',
    title_font=dict(size=16),
    xaxis=dict(title_font=dict(size=12)),
    yaxis=dict(title_font=dict(size=12)),
    margin=dict(l=10, r=10, t=30, b=10)
)


save_plot_as_html(fig10, "Paid Free Graph 10.html","Finding outlier with the help of Boxplot in comparision of paid vs free apps")


# In[61]:


plot_containers_split=plot_containers.split("<div>")


# In[62]:


if len(plot_containers_split)>1:
    final_plot=plot_containers_split[-2]+"</div>"
else:
    final_plot=plot_container


# In[63]:


dashboard_html = """
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Play Store Review Analytics</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #333;
            color: #fff;
            margin: 0;
            padding: 0;
        }}

        .header {{
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
            background-color: #444;
        }}

        .header img {{
            margin: 0 10px;
            height: 50px;
        }}

        .container {{
            display: block;
            flex-wrap: wrap;
            justify-content: center;
            padding: 20px;
        }}

        .plot-container {{
            border: 2px solid #555;
            margin: 10px;
            padding: 10px;
            width: {plot_width}px;
            height: {plot_height}px;
            overflow: hidden;
            position: relative;
            cursor: pointer;
        }}

        .insights {{
            display: block;
            position: absolute;
            right: 10px;
            top: 10px;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 5px;
            border-radius: 5px;
            color: #fff;
        }}

        .plot-container:hover .insights {{
            display: block;
        }}
    </style>

    <script>
        function openPlot(filename) {{
            window.open(filename, '_blank');
        }}
    </script>
</head>

<body>
    <div class="header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Logo_2013_Google.png/800px-Logo_2013_Google.png" 
             alt="Google Logo">
        <h1>Google Play Store Review Analytics</h1>
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Google_Play_Store_badge_EN.svg/1024px-Google_Play_Store_badge.svg.png" 
             alt="Google Play Store Badge" style="height: 50px;">
    </div>

    <div class="container">
        {plots}
    </div>
</body>

</html>
"""



# In[64]:


final_html = dashboard_html.format(plots=plot_containers, plot_width=plot_width, plot_height=plot_height)


# In[65]:


dashboard_path=os.path.join(html_files_path,"web page.html")


# In[66]:


with open("dashboard_path", "w",encoding="utf-8") as f:
    f.write(final_html)


# In[67]:


webbrowser.open('file://'+os.path.realpath(dashboard_path))


# In[68]:


merge_df.head(10)


# In[69]:


pip install wordcloud


# In[70]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


five_star_reviews = user_df[user_df['Sentiment Score'] >= 0.8]['Translated_Review']


text = ' '.join(five_star_reviews.astype(str))


stopwords_list = set(STOPWORDS)
app_names = set(user_df['App'].unique())
stopwords_list.update(app_names)

wordcloud = WordCloud(width=800, height=400,
                      background_color='white',
                      stopwords=stopwords_list,
                      min_font_size=10).generate(text)


plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[71]:


merge_df['Sentiment Score']=merge_df['Translated_Review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])


# In[72]:


merge_df.head(5)


# In[73]:


#Now focusing on app catgory: Health and Fitness
five_star_Health = merge_df[merge_df['Category']=="Health & Fitness"][merge_df['Sentiment Score'] >= 0.8]['Translated_Review']

text2 = ' '.join(five_star_Health.astype(str))

stopwordsHF = set(STOPWORDS)
app_names = set(user_df['App'].unique())
stopwordsHF.update(app_names)

wordcloud = WordCloud(width=800, height=400,
                      background_color='white',
                      stopwords=stopwordsHF,
                      min_font_size=10).generate(text)


plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# In[ ]:




