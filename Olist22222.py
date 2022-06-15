
from curses import color_content
from xml.etree import cElementTree
import requests
import zipfile
import sqlite3
import pandas as pd
import streamlit as st 
from datetime import datetime
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from googletrans import Translator
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.corpus import re
from wordcloud import WordCloud 
from PIL import Image
from plotly.subplots import make_subplots
from streamlit_multipage import MultiPage
import squarify
st.set_page_config(layout = "wide")
st.header("Purpose : improve Olist customer's satisfaction")
def reviw_class(x): 
  if x < 4:
    return 0
  if x >= 4: 
    return 1 
orders= pd.read_csv("orders.csv")
translation =pd.read_csv("product_category_name_translation.csv")
order_items_dataset = pd.read_csv("order_items_dataset.csv")
customers_dataset= pd.read_csv("customers_dataset.csv")
products_dataset= pd.read_csv("products_dataset.csv")
order_reviews_dataset= pd.read_csv("order_reviews_dataset.csv")
order_payments_dataset= pd.read_csv("order_payments_dataset.csv")
sellers_dataset = pd.read_csv("sellers_dataset.csv")
tab1 = pd.merge(order_items_dataset, orders, on="order_id")
tab2 = pd.merge(customers_dataset, tab1, on="customer_id")
tab3 = pd.merge(tab2, order_reviews_dataset, on = "order_id")
tab4 = tab3.drop_duplicates(subset=['order_id', 'order_item_id'], keep='last')
tab5= tab4.reset_index()
tab6= pd.merge(tab5, sellers_dataset, on= "seller_id")

tab6=tab6[tab6["order_status"] == "delivered"]
tab6["review_class1"] = tab6["review_score"].apply(reviw_class)

tab6["review_class2"] = tab6["review_class1"]
tab7 = pd.merge(tab6, products_dataset, on="product_id")
tab8 = pd.merge(tab7, sellers_dataset, on= "seller_id")
tab9 = pd.merge(tab8, order_payments_dataset, on= "order_id")

imagex = Image.open('olistlogo2.png')

#code pour insérer les premiers metrics des CA : 
tab9["order_purchase_timestamp"] = pd.to_datetime(tab9["order_purchase_timestamp"], format='%Y-%m-%d')
tab9["Year"]= tab9["order_purchase_timestamp"].dt.year
tab9["Month"]= tab9["order_purchase_timestamp"].dt.month

col1, col2, col3, col4= st.columns(4)

CA =tab9.groupby(["Year"])[["payment_value"]].sum()


col1.metric(label = "2016 Turnover", value= f"{CA.iloc[0,0]:,} R$")
col2.metric(label = "2017 Turnover", value= f"{CA.iloc[1,0]:,} R$")
col3.metric(label = "2018 Turnover", value= f"{CA.iloc[2,0]:,} R$",delta= f"21.12 %")
col4.image(imagex,  width= 120)


#code pour faire un graphique de l'évolution de l'activité de l'entreprise avec un filtre
st.header ("Olist Actvitiy Evolution")
year_filter1 = st.selectbox('select a year', pd.unique(tab9["Year"]))
tab9 = tab9[tab9["Year"] == year_filter1]
tab9['payment_value1']= tab9["payment_value"].map('{:,.3f}'.format)
CA2 =tab9.groupby(["Year", "Month"]).agg({'payment_value': 'sum'})
CA2= CA2.reset_index()

figx1 = px.bar(CA2, x = "Month", y= "payment_value", title = "Monthly turnover", color="payment_value")
figx1.update_layout(
    
    coloraxis_colorbar=dict(
    title="Turnover"),
    xaxis_title="Month",
    yaxis_title="Turnover",
   
    
    font=dict(size=14))

st.plotly_chart(figx1, use_container_width= True)
#here starts the game
or_rev = pd.merge(order_reviews_dataset, orders, on= "order_id")
or_rev1 = or_rev[or_rev["order_status"]== "delivered"][["order_id", "review_id", "review_score", "review_comment_title", "review_comment_message", "order_status", "order_purchase_timestamp", "order_estimated_delivery_date", "order_delivered_customer_date"]]
or_rev2 = or_rev1.drop_duplicates(subset=['order_id'], keep='last')

or_rev2["order_purchase_timestamp"] = pd.to_datetime(or_rev2["order_purchase_timestamp"], format='%Y-%m-%d')
or_rev2["order_purchase_timestamp2"] = or_rev2["order_purchase_timestamp"].apply(lambda x: datetime.strftime(x, '%Y-%m'))
or_rev2["order_purchase_timestamp3"] = or_rev2["order_purchase_timestamp2"].apply(lambda x: datetime.strptime(x, '%Y-%m'))

or_rev2["Year"]= or_rev2["order_purchase_timestamp3"].dt.year
or_rev2["Month"]= or_rev2["order_purchase_timestamp3"].dt.month

or_rev3 = or_rev2
#creation du filtre par année 
st.header ("The number of orders and reviews issued on a monthly basis")
year_filter = st.selectbox('select a year', pd.unique(or_rev3["Year"]))
or_rev3 = or_rev3[or_rev3["Year"] == year_filter]
Monthly_orders = or_rev3.groupby(["Month"]).agg({'order_id': 'count', 'review_comment_message':'count'})
Monthly_orders= Monthly_orders.reset_index()

#st.write(Monthly_orders)

# ici je plot les premiers graphs



figx2 = make_subplots(1,1)

figx2.add_trace(go.Bar(x=Monthly_orders['Month'], y=Monthly_orders['order_id'],
                     name='number of orders',
                     marker_color = 'turquoise',
                     
                     marker_line_color='rgb(8,48,107)',
                     marker_line_width=2))


figx2.add_trace(go.Scatter(x=Monthly_orders['Month'], y=Monthly_orders['review_comment_message'], line=dict(color='red'), name='number of reviews'))


st.plotly_chart(figx2, use_container_width= True)



or_rev2["review_class"] = or_rev2["review_score"].apply(reviw_class)




# figures pour les reviews scores

bad_review_ev = or_rev2.pivot_table(values = "review_score", index= ["Year","Month"], columns="review_class", aggfunc= "count")

bad_review_ev1 = bad_review_ev.reset_index()




col1, col2, col3 = st.columns(3)
#fig3 = px.bar(bad_review_ev1, x = "Month" , y= [0,1], color_discrete_sequence =['salmon', '#1f77b4'])

#fig3.update_layout(
    #title="Monthly evolution of review classes",
    #xaxis_title="Month",
    #yaxis_title="nbr of review scores",
    #legend_title="Review class",
    #font=dict(size=14))

#col1.plotly_chart(fig3)
st.header ("The distribution of review scores")
col1, col2, col3 = st.columns(3)



col1.metric(label = "2016 penalties", value= f"{6930:,} R$")
col2.metric(label = "2017 penalties", value= f"{670140:,} R$")
col3.metric(label = "2018 penalties", value= f"{1080190 :,} R$")


sizesx1 = round(or_rev2['review_score'].value_counts(normalize= True)*100,2)

labels1 = "score = 5", "score = 4", "score = 3", "score = 2", " score = 1"

figx3 = px.pie(or_rev2, values= sizesx1, names= labels1, title='Distribution of review scores', hole=.3, color = sizesx1, color_discrete_map={'Thur':'lightcyan',
                                 'Fri':'cyan',
                                 'Sat':'royalblue',
                                 'Sun':'darkblue'})
col2.plotly_chart(figx3, use_container_width = True)
st.header("What do the reviews say ?")
col1, col2 = st.columns(2)

image1 = Image.open('worldcloud_good.png')

col1.image(image1, caption='most common words in good reviews')

image2 = Image.open('wordcloud_bad.png')

col2.image(image2, caption='most common words in bad reviews')

# à partir d'ici je plot les pies 
# mais d'abord je fais les pourcentages de commandes à l'heure ou en retard


or_rev2["order_delivered_customer_date"]= pd.to_datetime(or_rev2["order_delivered_customer_date"])
or_rev2["order_estimated_delivery_date"] =pd.to_datetime(or_rev2["order_estimated_delivery_date"])

or_rev2["wait_time"]= round((or_rev2["order_estimated_delivery_date"] - or_rev2["order_delivered_customer_date"]) /np.timedelta64(24, "h"),2)

late_orders =or_rev2[or_rev2["wait_time"] < 0]
late_orders_count = or_rev2["order_id"][or_rev2["wait_time"] < 0].count()
early_orders_count =or_rev2["order_id"][or_rev2["wait_time"] > 0].count()
sum_of_orders = or_rev2["order_id"].count()
col1, col2 = st.columns(2)
per_late = round((late_orders_count * 100)/sum_of_orders,2)
per_early =round((early_orders_count * 100)/sum_of_orders, 2)

col2.metric(label= "% of orders which arrived late", value= per_late)
col1.metric(label= "% of orders which arrived early", value= per_early )
sizes = round(late_orders['review_score'].value_counts(normalize= True)*100,2)
labels = "score = 1", "score = 2", "score = 3", "score = 4", " score = 5"

col1, col2 = st.columns(2)

fig4 = px.pie(late_orders, values= sizes, names= labels, title='Distribution of review scores of late orders', hole=.3, color = sizes, color_discrete_map={'Thur':'lightcyan',
                                 'Fri':'cyan',
                                 'Sat':'royalblue',
                                 'Sun':'darkblue'})

col1.plotly_chart(fig4, use_container_width = True)

early_orders =or_rev2[or_rev2["wait_time"] > 0]
sizes1 = round(early_orders['review_score'].value_counts(normalize= True)*100,2)
sizes1= sizes1.sort_values(ascending = True )
labels1 = "score = 1", "score = 2", "score = 3", "score = 4", " score = 5"
fig5 = px.pie(early_orders, values= sizes1, names= labels1, title='Distribution of review scores of early orders', hole=.3, color = sizes, color_discrete_map={        
                                 'Fri':'cyan',
                                 'Sun':'darkblue',
                                 'Sat':'royalblue',
                                 
                                 'Thur':'lightcyan'})
col2.plotly_chart(fig5, use_container_width = True)


#code pour importer mes deux wordcloud

##
#
#
#
#ici commence le code pour les sellers 

seller_classes = tab6.pivot_table(values = "review_class1", index= "seller_id", aggfunc= "count", columns = "review_class2")

seller_classes2= seller_classes.dropna()

seller_classes["Nbr of Orders"]= tab6.groupby("seller_id").agg({"order_id": "count"})


seller_classes2= seller_classes.dropna()

seller_classes2["percentage_of_bad_reviews"] = (seller_classes2[0] * 100)/ (seller_classes2[0] + seller_classes2[1])

seller_classes20 = seller_classes2[seller_classes2["Nbr of Orders"] > 20]

fig7 = px.scatter(seller_classes20, x = "Nbr of Orders" , y="percentage_of_bad_reviews", title= "sellers by number of orders and % of bad reviews"
  , color_discrete_sequence=['#AB63FA'] )

fig7.update_layout(
    
    xaxis_title="nb of orders",
    yaxis_title="% of bad scores"
    
    )      
st.plotly_chart(fig7, use_container_width = True)



fig7.update_traces(textposition="top center")



















tab10 = pd.merge(tab9, translation, on="product_category_name", how = "inner")

tab10["review_class1"] = tab10["review_score"].apply(reviw_class)

tab10["review_class2"] = tab10["review_class1"]

tab10.drop("Year",inplace= True, axis = 1)

tab11 = tab10 



products_classes = tab11.pivot_table(values = "review_class1", index= "product_category_name_english", aggfunc= "count", columns = "review_class2")


products_classes["Nbr of Orders"]= round(tab11.groupby("product_category_name_english").agg({"order_id": "count"}), 2)


products_classes2= products_classes.dropna()

products_classes2["percentage_of_bad_reviews"] = round((products_classes2[0] * 100)/ (products_classes2[0] + products_classes2[1]),2)


products_classes20 = products_classes2[products_classes2["Nbr of Orders"] > 10].sort_values(by= "Nbr of Orders", ascending = False)

products_classes20 


tab11["order_id2"]= tab11["order_id"]



tab_seller = tab11.pivot_table(values = ("review_class1"), index= ("product_category_name_english", "seller_id"), aggfunc= "count", columns = ("review_class2"))



tab_seller1 = tab_seller.dropna()
tab_seller1 = tab_seller1.reset_index()
category_filter = st.selectbox('select a product category', pd.unique(tab_seller1["product_category_name_english"]))
tab_seller1 = tab_seller1[tab_seller1["product_category_name_english"] == category_filter]



fig88 = px.scatter(tab_seller1, x = 1 , y= 0, title= "sellers by number of good and bad reviews"
  , color_discrete_sequence=['white'], labels = {0 : "seller", 1: "seller"})

fig88.update_traces(
        marker_size=30,
        marker_line=dict(width=9),
        selector=dict(mode='markers',
        )
    )
fig88.update_layout(
    
    xaxis_title="number of Good scores",
    yaxis_title="number bad scores",
    
    
    )   
st.plotly_chart(fig88, use_container_width = True)


st.header("Prices distribution")
figXY = px.histogram(tab10, x="price")
figXY.update_layout(
  
    xaxis_title="Price",
    yaxis_title="Distribution",
   
    
    font=dict(size=14))

st.plotly_chart(figXY, use_container_width= True)


st.header("Sellers segmentation")








tab_rfm = tab10[["seller_id","payment_value", "customer_unique_id", "order_purchase_timestamp"]]
tab_rfm["order_purchase_timestamp_timed"] = pd.to_datetime(tab_rfm["order_purchase_timestamp"], format='%Y-%m-%d %H:%M:%S')
tab_rfm = tab_rfm.groupby("seller_id").agg({"customer_unique_id" : "count", "payment_value" : "sum", "order_purchase_timestamp_timed" : "max"})\
  .sort_values(by = "order_purchase_timestamp_timed", ascending = False)
rfm1 = tab_rfm.iloc[:1030, :]
rfm2 = tab_rfm.iloc[1031:2060, :]
rfm3 = tab_rfm.iloc[2061:3090, :]
rfm1["Recency"] = 3
rfm2["Recency"] = 2
rfm3["Recency"] = 1
frames = [rfm1, rfm2, rfm3]
RFM_Recency = pd.concat(frames)
rfm1 = tab_rfm.iloc[:1030, :]
rfm2 = tab_rfm.iloc[1031:2060, :]
rfm3 = tab_rfm.iloc[2061:3090, :]
rfm1["Recency"] = 3
rfm2["Recency"] = 2
rfm3["Recency"] = 1
frames = [rfm1, rfm2, rfm3]
RFM_Recency = pd.concat(frames)
RFM_Recency= RFM_Recency.sort_values(by="customer_unique_id", ascending = False)
rfm1_1 = RFM_Recency.iloc[:1030, :]
rfm2_2 = RFM_Recency.iloc[1031:2060, :]
rfm3_3 = RFM_Recency.iloc[2061:3090, :]
rfm1_1["Frequecy"] = 3
rfm2_2["Frequecy"] = 2
rfm3_3["Frequecy"] = 1
frames = [rfm1_1, rfm2_2, rfm3_3]
RFM_Freq = pd.concat(frames)
RFM_Freq= RFM_Freq.sort_values(by="payment_value", ascending = False)
rfm1_1_1 = RFM_Freq.iloc[:1030, :]
rfm2_2_2 = RFM_Freq.iloc[1031:2060, :]
rfm3_3_3 = RFM_Freq.iloc[2061:3090, :]
rfm1_1_1["Monetary"] = 3
rfm2_2_2["Monetary"] = 2
rfm3_3_3["Monetary"] = 1
frames_2 = [rfm1_1_1, rfm2_2_2, rfm3_3_3]
RFM_Mon = pd.concat(frames_2)
RFM_Mon["RFM_score"] = (RFM_Mon["Recency"] * 100) + (RFM_Mon["Frequecy"] * 10) + (RFM_Mon["Monetary"] * 1)
def RFM_score(x):
  if x == 333:
    return "Top Sellers"
  elif x == 332 or x == 331 or x == 323 or x == 313:
    return "Potential champions"
  elif x== 321 or x== 322 or x == 311 or x == 312:
    return "In between 3d group"
  elif x == 233 :
    return "Ex loyals/ need a attention/ promotion"
  elif x == 223 or x == 213 or x == 212 or x == 231  or x == 232 or x == 211 or x == 221 or x == 222 :
    return "Average Sellers"
  elif x == 132 or x == 123 or x == 113 or x == 133 :
    return "They haven't sold for a while"
  elif x == 111 or x == 112 or x == 121 or x == 122 or x == 131:
    return "Sellers who lose customers"
    
RFM_Mon["Sellers_Seg"] = RFM_Mon["RFM_score"].apply(RFM_score)
rfm_level_agg = RFM_Mon.groupby('Sellers_Seg').agg({
    'Recency': 'mean',
    'Frequecy': 'mean',
    'Monetary': ['mean', 'count']
}).round(1)
print(rfm_level_agg)
rfm_level_agg.columns = ['RecencyMean','FrequecyMean','MonetaryMean', 'Count']
#fig = plt.gcf()
#ax = fig.add_subplot()
#fig.set_size_inches(16, 9)
#figX12 = squarify.plot(sizes=rfm_level_agg['Count'], 
              #label=['Average Sellers',
                     #'Sellers who lose customers',
                     #'Top Sellers',
                    # 'Ex loyals/ need a attention/ promotion',
                    # 'in between 3d group',
                    # "They haven't sold for a while", 
                    # "Potential champions"], alpha=.6)

#plt.title("RFM Segments",fontsize=18,fontweight="bold")
#plt.axis('off')
#plt.show();
#st.set_option('deprecation.showPyplotGlobalUse', False)
#st.pyplot();
RFM_norm= RFM_Mon.groupby("Sellers_Seg").count()\
.sort_values(by ="customer_unique_id", ascending= False)[["customer_unique_id"]]

RFM_norm["percent"] = round((RFM_norm['customer_unique_id'] / RFM_norm['customer_unique_id'].sum()) * 100, 2)

RFM_norm.groupby(['Sellers_Seg']).sum()

sizes33 = round(RFM_Mon['Sellers_Seg'].value_counts(normalize= True)*100,2)
labels33 = "Average sellers", "Ex loyals", "above average", "potential champions", "sellers who lose cutomers"


##fig333 = px.pie(RFM_Mon, values= sizes33, names= labels33, title='Distribution of review scores of late orders', hole=.3, color = sizes, color_discrete_map={'Thur':'lightcyan',
                                 #'Fri':'cyan',
                                 #'Sat':'royalblue',
                                 ##'Sun':'darkblue'})

#st.plotly_chart(fig333, use_container_width = True)


tab_seg = tab10[["seller_id","payment_value", "customer_unique_id", "review_score", "order_delivered_customer_date", "order_estimated_delivery_date"]]
tab_seg["order_delivered_customer_datetimed"] = pd.to_datetime(tab_seg["order_delivered_customer_date"], format='%Y-%m-%d %H:%M:%S')
tab_seg["order_estimated_delivery_datetimed"] = pd.to_datetime(tab_seg["order_estimated_delivery_date"], format='%Y-%m-%d %H:%M:%S')
tab_seg["retard"]= tab_seg["order_delivered_customer_datetimed"]-tab_seg["order_estimated_delivery_datetimed"]
RFMX1=RFM_Mon[["Sellers_Seg"]]
RFM_S = RFMX1.merge(tab_seg, on="seller_id", how = "right")
RFM_S[["retard"]].mean()
RFM_S.groupby("Sellers_Seg").agg({"review_score": ["mean", "count", "min","max"], "retard": ["mean", "count", "min", "max"]})

commandes_en_retard = RFM_S[(RFM_S["retard"]/ np.timedelta64(24, "h")) > 0]
commandes_en_avance = RFM_S[(RFM_S["retard"]/ np.timedelta64(24, "h")) < 0]

commandes_en_retard.groupby("Sellers_Seg").agg({"retard": ["mean", "count", "min", "max"]})
graph_retard = commandes_en_retard.groupby("Sellers_Seg").agg({"retard": "mean"})
graph_retard["retard"]= graph_retard["retard"].dt.days
image33 = Image.open('sellers.png')
col1, col2 = st.columns(2)
RFM_graph = pd.read_csv("filename.csv")
col1.image(image33, width= 500 )
col2.write(RFM_graph)


st.header("Customers segmentation")

col1, col2, col3 = st.columns(3)
image34 = Image.open('customers.png')
col2.image(image34, width= 600, use_column_width=True )


#st.header("customers and sellers segmentation analysis")
RFM_graph2 = pd.read_csv("RFM111.csv")
#st.write(RFM_graph2)

st.header("Recommandations")

st.write("Prospect et recruit new sellers to diversify the range of suggested products to the customers") 
st.write("Work on the delays, either reduce the delays or declare late deliveries to the customers from the start") 
st.write("Reajust the subscription price to the sellers, to encourage good sellers ") 
st.write("Suggest a subscription to the customers and offer reduced prices to the customers who buy frequently") 
st.write("Compensate for late delays to avoid getting bad scores") 
st.write("Implement  : 1. a hotline to manage delays and value the customer.")
st.write("2) a satisfaction survey to precisely collect the reasons for dissatisfaction.")   
st.write("3) A qualitative study with expert customers to study the customer journey")
st.write("4) Make an analysis of the the logistics strategy: identify critical points that can be improved.")
col1,col2,col3= st.columns(3)
col2.title("Thank you for your")
col1,col2,col3= st.columns(3)
col2.title("         attention!")