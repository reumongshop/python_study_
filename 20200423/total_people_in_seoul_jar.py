# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:23:21 2020

@author: USER
"""

import pandas as pd
import folium
import webbrowser
 

total_people_geo = 'c:/python_data/TL_SCCO_SIG_WGS84.json'

total_people_data = pd.read_csv('c:/python_data/Total_People_2018.csv', encoding = 'euc-kr')
print(total_people_data)

m = folium.Map(location=[36, 127], tiles="OpenStreetMap", zoom_start=7)

m.choropleth(geo_data=total_people_geo,
             name='choropleth',
             data=total_people_data,
             columns=['Code', 'Population'],
             key_on='feature.properties.SIG_CD',
             fill_color='YlGn',
             fill_opacity=0.7,
             line_opacity=0.5,
             legend_name='Population Rate (%)'
             )

folium.LayerControl().add_to(m)
 
m.save('folium_kr.html')
webbrowser.open_new("folium_kr.html")


# =============================================================================
# state_unemployment = 'c:/python_data/02. folium_US_Unemployment_Oct2012.csv'
# state_data = pd.read_csv(state_unemployment)
# state_data.head()
# 
# state_geo = "c:/python_data/02. folium_us-states.json"
# 
# map = folium.Map(location=[36, 127],
#                  )
# =============================================================================
