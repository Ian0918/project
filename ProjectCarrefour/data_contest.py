#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import numpy as np

import seaborn as sns


# In[6]:


orders= pd.read_csv( 'Downloads/file_62aace1a6227f.csv')


# In[5]:


ordersfinal= pd.read_csv( 'ordersfinal.csv')


# In[61]:


ordersRF= pd.read_csv( 'Downloads/file_62aace1a6227f 改.csv')
ordersRF['order_date'] = pd.to_datetime(ordersRF['order_date'])


# In[76]:


ordersRFM.sort_values('order_date').reset_index(drop=True)


# In[81]:


ordersfianl=pd.concat([ordersRFM.sort_values('order_date').reset_index(drop=True),ordersRF[['CPI','confirmed']]], axis=1)
ordersfianl.to_csv('ordersfinal.csv')


# In[68]:


order=pd.merge(ordersRFM,ordersRF[['order_date','CPI','confirmed']], how='inner', left_on='order_date', right_on='order_date')
order


# In[70]:


ordersRFM.merge(ordersRF[['order_date','CPI','confirmed']])


# In[6]:


orders.drop(labels=['Unnamed: 12','Unnamed: 13','Unnamed: 14'], axis=1, inplace=True)


# In[7]:


len(orders['customer'].unique())


# In[8]:


len(orders['product'].unique())


# In[9]:


pd.to_datetime(orders['order_date']).max()


# In[10]:


orders['order_date date']=orders['order_date'][0].split()[0]


# In[11]:


orders['order_date time']=orders['order_date'][0].split()[1]


# In[12]:


orders['order_date'] = pd.to_datetime(orders['order_date'])


# In[13]:


import datetime


# In[14]:


# 獲得起始結束時間範圍内的數據
def time_period(start:str, end:str, df):
  '''
  start end 格式：%Y-%m-%d %H:%M:%S
  是根據"OrderFinishDateTime"來選擇時間區間
  '''
  s_date = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S').date()
  e_date = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S').date()
  return df[(df.order_date > pd.Timestamp(s_date)) & (df.order_date < pd.Timestamp(e_date))]


# In[15]:


# 獲得起始結束時間範圍内的數據
def time_period(start:str, end:str, df):
  '''
  start end 格式：%Y-%m-%d %H:%M:%S
  是根據"OrderFinishDateTime"來選擇時間區間
  '''
  s_date = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S').date()
  e_date = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S').date()
  return df[(df.order_date > pd.Timestamp(s_date)) & (df.order_date < pd.Timestamp(e_date))]


# In[16]:


time_period('2020-05-12 00:00:00', '2021-12-31 00:00:00', orders)


# In[17]:


recency = pd.DataFrame(orders.groupby('customer')['order_date'].max())
recency


# In[18]:


recency['R'] = pd.Timestamp("2021-12-23 03:33:00") - recency['order_date']
x=pd.Timestamp("2021-12-23 03:33:00")
recency


# In[19]:


frequency = pd.DataFrame(orders.groupby('customer')['id'].count())
frequency


# In[20]:


monetary = pd.DataFrame(orders.groupby('customer')['sales_price'].sum())
monetary


# **產生RFM**

# In[21]:


clean_data = pd.merge(recency, frequency, how='inner', left_on='customer', right_on='customer')
clean_data


# In[22]:


clean_data = pd.merge(clean_data, monetary, how='inner', left_on='customer', right_on='customer')
clean_data


# In[23]:


clean_data = clean_data[['R', 'id', 'sales_price']]
clean_data.columns = ['recency', 'frequency', 'monetary']
#clean_data.reset_index(inplace=True)


# In[24]:


clean_data.reset_index(inplace=True)


# In[25]:


clean_data


# In[26]:


clean_data["recency"] = clean_data["recency"].dt.days


# In[27]:


orders=pd.merge(orders,clean_data, how='inner', left_on='customer', right_on='customer')


# In[28]:


orders


# In[29]:


print(np.percentile(clean_data['monetary'],50,interpolation='midpoint'))


# In[30]:


clean_data['recency'].sum()/8


# In[31]:


for i in range(0,7):
    if clean_data['monetary'][i]<11380.0:
        clean_data['monetary'][i]=0
    else:
        clean_data['monetary'][i]=1
for i in range(0,7):
    if clean_data['frequency'][i]<99.875:
        clean_data['frequency'][i]=0
    else:
        clean_data['frequency'][i]=1   
for i in range(0,7):
    if clean_data['recency'][i]<250.75:
        clean_data['recency'][i]=0
    else:
        clean_data['recency'][i]=1
clean_data.iloc[7,1]=1


# In[32]:


clean_data.iloc[7,2]=0
clean_data


# In[33]:


clean_data.iloc[7,3]=1
clean_data['RFMlevel']=['rH fL','rH fL','rL fL','rL fH','rL fL','rH fL','rL fH','rH fL']
clean_data


# In[34]:


orders=pd.merge(orders,clean_data, how='inner', left_on='customer', right_on='customer')


# In[35]:


ordersRFM=orders


# In[36]:


orders.groupby('order_date')['sales_price'].sum()


# In[37]:


x=orders[orders['customer']=='00113cb1-293b-4c73-8844-4ca901c819ab']
y=orders[orders['customer']=='003c1701-7951-41f7-8e3e-7c102daa28a0']
z=orders[orders['customer']=='0053e832-3011-4464-9157-955cd5b1fb76']
q=orders[orders['customer']=='00a0c55f-1f16-4475-a7a3-82c5dd6c0ff4']
w=orders[orders['customer']=='00eedbf8-1784-4e80-8ce3-a11ef8198c87']
e=orders[orders['customer']=='01220559-7825-428b-b0dd-abe76688c3bb']
r=orders[orders['customer']=='01e377d9-2b83-4c9f-96ba-ed468c510eb3']
t=orders[orders['customer']=='020457be-42c6-4074-b15c-ddae166d297a']


# In[38]:


x.groupby('order_date')['sales_price'].sum()


# In[39]:


import matplotlib.pyplot as plt


# In[40]:


plt.plot(x.groupby('order_date')['sales_price'].sum().index,x.groupby('order_date')['sales_price'].sum().values)
plt.plot(y.groupby('order_date')['sales_price'].sum().index,y.groupby('order_date')['sales_price'].sum().values)
plt.plot(z.groupby('order_date')['sales_price'].sum().index,z.groupby('order_date')['sales_price'].sum().values)
plt.plot(q.groupby('order_date')['sales_price'].sum().index,q.groupby('order_date')['sales_price'].sum().values)
plt.plot(w.groupby('order_date')['sales_price'].sum().index,w.groupby('order_date')['sales_price'].sum().values)
plt.plot(e.groupby('order_date')['sales_price'].sum().index,e.groupby('order_date')['sales_price'].sum().values)
plt.plot(r.groupby('order_date')['sales_price'].sum().index,r.groupby('order_date')['sales_price'].sum().values)
plt.plot(t.groupby('order_date')['sales_price'].sum().index,t.groupby('order_date')['sales_price'].sum().values)


# In[41]:


cluster1=orders[orders['customer']=='RFMlevel']


# In[99]:


# create model
model = Sequential()


# In[113]:


model.add(Dense(10, input_dim=3, activation='relu')) 
model.add(Dense(3,  activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# In[ ]:


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # Fit the model
model.fit(X, Y, 150, 10)
# evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[68]:


orders['customer'].unique()


# In[69]:


orders[orders['customer']=='003c1701-7951-41f7-8e3e-7c102daa28a0']['order_date'].unique()


# In[72]:


select4=orders[orders['customer']=='003c1701-7951-41f7-8e3e-7c102daa28a0'][orders['order_date']=='2020/12/17 13:05']
select5=orders[orders['customer']=='003c1701-7951-41f7-8e3e-7c102daa28a0'][orders['order_date']=='2021/3/8 02:59']
select6=orders[orders['customer']=='003c1701-7951-41f7-8e3e-7c102daa28a0'][orders['order_date']=='2020/11/3 12:01']


# In[75]:


np.intersect1d(select5["product"], select6["product"], assume_unique=False) 


# In[52]:


orders[orders['customer']=='00113cb1-293b-4c73-8844-4ca901c819ab']


# In[63]:


select3=orders[orders['customer']=='00113cb1-293b-4c73-8844-4ca901c819ab'][orders['order_date']=='2020/7/3 10:35']
select3


# In[46]:


plt.figure(figsize=(4,6))    # 顯示圖框架大小

labels = select["product"]      # 製作圓餅圖的類別標籤
#separeted = (0, 0, 0.3, 0, 0.3)                  # 依據類別數量，分別設定要突出的區塊
size = select['quantity']                       # 製作圓餅圖的數值來源

plt.pie(size,                           # 數值
        labels = labels,                # 標籤
        autopct = "%1.1f%%",            # 將數值百分比並留到小數點一位
            # 設定分隔的區塊位置
        pctdistance = 0.6,              # 數字距圓心的距離
        textprops = {"fontsize" : 12},  # 文字大小
        shadow=True)                    # 設定陰影

plt.axis('equal')                                          # 使圓餅圖比例相等
plt.title("2020/7/17 10:13", {"fontsize" : 18})  # 設定標題及其文字大小
                                  # 設定圖例及其位置為最佳


# In[51]:


plt.figure(figsize=(4,6))    # 顯示圖框架大小


labels = select1["product"]      # 製作圓餅圖的類別標籤
#separeted = (0, 0, 0.3, 0, 0.3)                  # 依據類別數量，分別設定要突出的區塊
size = select1['quantity']                       # 製作圓餅圖的數值來源

plt.pie(size,                           # 數值
        labels = labels,                # 標籤
        autopct = "%1.1f%%",            # 將數值百分比並留到小數點一位
            # 設定分隔的區塊位置
        pctdistance = 0.6,              # 數字距圓心的距離
        textprops = {"fontsize" : 12},  # 文字大小
        shadow=True)                    # 設定陰影

plt.axis('equal')                                          # 使圓餅圖比例相等
plt.title("2020/8/31 11:45", {"fontsize" : 18})  # 設定標題及其文字大小
                                  # 設定圖例及其位置為最佳


# In[54]:


plt.figure(figsize=(4,6))    # 顯示圖框架大小


labels = select2["product"]      # 製作圓餅圖的類別標籤
#separeted = (0, 0, 0.3, 0, 0.3)                  # 依據類別數量，分別設定要突出的區塊
size = select2['quantity']                       # 製作圓餅圖的數值來源

plt.pie(size,                           # 數值
        labels = labels,                # 標籤
        autopct = "%1.1f%%",            # 將數值百分比並留到小數點一位
            # 設定分隔的區塊位置
        pctdistance = 0.6,              # 數字距圓心的距離
        textprops = {"fontsize" : 12},  # 文字大小
        shadow=True)                    # 設定陰影

plt.axis('equal')                                          # 使圓餅圖比例相等
plt.title("2020/7/10 10:42", {"fontsize" : 18})  # 設定標題及其文字大小
                                  # 設定圖例及其位置為最佳


# In[67]:


np.intersect1d(select["product"], select3["product"], assume_unique=False) 


# In[110]:


orders


# In[7]:


dp=orders.drop_duplicates(subset=['product'])
for i in range(0,482):
    if dp['quantity'].iloc[i]!=1:
        dp['sales_price'].iloc[i]=dp['sales_price'].iloc[i]/dp['quantity'].iloc[i]
dp.rename(columns = {'sales_price':'persales'}, inplace = True)


# In[8]:


orders=pd.merge(orders
         ,dp)


# In[9]:


dp=orders.drop_duplicates(subset=['product'])
dp['quantity'].loc[1]!=1


# In[10]:


dp['quantity'].iloc[20]


# In[11]:


dp=orders.drop_duplicates(subset=['product'])


# In[14]:


ordersfinal['persales']=orders['persales']


# In[ ]:


print(np.percentile(orders['persales'],20,interpolation='midpoint'))
print(np.percentile(orders['persales'],40,interpolation='midpoint'))
print(np.percentile(orders['persales'],60,interpolation='midpoint'))
print(np.percentile(orders['persales'],80,interpolation='midpoint'))
print(np.percentile(orders['persales'],100,interpolation='midpoint'))


# In[16]:


def f1(row):
    if row <=38:
        val = 1
    elif row <= 59.0:
        val = 2
    elif row <= 83.0:
        val = 3
    elif row <= 131.0:
        val = 4
    else:
        val = 5
    return val
    
ordersfinal['persalelevel'] = ordersfinal['persales'].apply(f1)
ordersfinal


# In[43]:


ordersRFM=orders


# In[18]:


ordersfinal.to_csv('ordersfinal.csv')


# In[41]:


ordersRFM['monetary']=orders['monetary']
ordersRFM


# In[44]:


ordersRFM


# In[69]:


#df_seg = df_seg.rename(columns = {'recency_cate':'近因'})
#df_seg = df_seg.rename(columns = {'frequency_cate':'頻率'})
recency_label =  [0,1]
frequency_label=[0,1]
g = sns.FacetGrid(ordersRFM, # 來源資料表
                  col="recency_y", # X資料來源欄位
                  row="frequency_y" ,  # Y資料來源欄位
                  col_order= recency_label,  # X資料順序
                  row_order= frequency_label, # Y資料順序
                  sharex=False,
            sharey=False,
                    size=2.2, aspect=1.6,
                  palette='Set1',  #畫布色調
                  margin_titles=True,
                  hue='customer'
                  )
g = g.map_dataframe(sns.barplot,'customer', 'monetary_x')
g = g.set_axis_labels('recency_y','frequency_y').add_legend()


# In[56]:


help(sns.FacetGrid)


# In[87]:


g = sns.FacetGrid(df_seg, # 來源資料表
                  col="近因", # X資料來源欄位
                  row="頻率" ,  # Y資料來源欄位
                    col_order= recency_label,  # X資料順序
                    row_order= frequency_label, # Y資料順序
                    sharex=False,
            sharey=False,
                    size=2.2, aspect=1.6,
                  palette='Set1',  #畫布色調
                    margin_titles=True,
                    hue='customer'
                  )
#小圖表部分
g = g.map(sns.barplot, 'gender' ,'顧客數量')
g = g.add_legend()


# In[71]:


#小圖表部分

g
#g.savefig("RFplot.png")


# In[ ]:




