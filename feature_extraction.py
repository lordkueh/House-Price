# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 17:34:04 2018

@author: bubu
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import geopy.distance as gpd


#importing dataset

dataset = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\resale_data1900-1999.csv')
dataset2 = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\resale_data 2000-2012.csv')
dataset3 = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\resale_data 2012-2014.csv')
dataset4 = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\resale_data 2015-.csv')

feature_columns_to_use = ['floor_area_sqm','resale_price']

feature_columns_to_use2 = ['floor_area_sqm','flat_type', 'flat_model', 'storey_range', 'resale_price']

# combined all the size vs price data into one dataset

dataset_house_size_vs_price = dataset[feature_columns_to_use].append(dataset2[feature_columns_to_use])
dataset_house_size_vs_price = dataset_house_size_vs_price.append(dataset3[feature_columns_to_use])
dataset_house_size_vs_price = dataset_house_size_vs_price.append(dataset4[feature_columns_to_use])

# combined flat type, flat model and resale_price to one dataset


dataset_house_type_vs_price = dataset[feature_columns_to_use2].append(dataset2[feature_columns_to_use2])
dataset_house_type_vs_price = dataset_house_type_vs_price.append(dataset3[feature_columns_to_use2])
dataset_house_type_vs_price = dataset_house_type_vs_price.append(dataset4[feature_columns_to_use2])





# =============================================================================
# extracting number of rooms converting to integer instead of string
# # LEGEND
# #   1 ROOM = 1
# #   2 ROOM = 2
# #   3 ROOM = 3
# #   4 ROOM = 4
# #   5 ROOM = 5
# #   6 ROOM = 6
# #   7 ROOM = 7
# #   8 ROOM = 8
# #   9 ROOM = 9
# #   EXECUTIVE = 10
# #   MULTI-GENERATION = 11
# =============================================================================

data = dataset_house_type_vs_price.iloc[:,1].values
data_ROOM_1 = [1 if element == '1 ROOM' else element for element in data]
data = data_ROOM_1
data_ROOM_2 = [2 if element == '2 ROOM' else element for element in data]
data = data_ROOM_2
data_ROOM_3 = [3 if element == '3 ROOM' else element for element in data]
data = data_ROOM_3
data_ROOM_4 = [4 if element == '4 ROOM' else element for element in data]
data = data_ROOM_4
data_ROOM_5 = [5 if element == '5 ROOM' else element for element in data]
data = data_ROOM_5
data_ROOM_6 = [6 if element == '6 ROOM' else element for element in data]
data = data_ROOM_6
data_ROOM_7 = [7 if element == '7 ROOM' else element for element in data]
data = data_ROOM_7
data_ROOM_8 = [8 if element == '8 ROOM' else element for element in data]
data = data_ROOM_8
data_ROOM_9 = [9 if element == '9 ROOM' else element for element in data]
data = data_ROOM_9
data_ROOM_10 = [10 if element == 'EXECUTIVE' else element for element in data]
data = data_ROOM_10
data_ROOM_11 = [11 if element == 'MULTI-GENERATION' else element for element in data]
data = data_ROOM_11


dataset_house_type_vs_price.iloc[:,1] = data

# =============================================================================
# # extracting type of house in the form of binary or integer
# # LEGEND
# #   improved = 1
# #   new generation = 2
# #   model A = 3
# #   model A2 = 4
# #   DBSS = 5
# #   standard = 6
# #   simplified = 7
# #   maisonette = 8
# #   MODEL A-MAISONETTE = 9
# #   premium apartment = 11
# #   apartment = 10
# #   type S1 = 12
# #   type S2 = 13
# #   adjoined flat = 14
# #   MULTI-GENERATION = 15
# #   TERRACE = 16
# #   2-ROOM = 17
# #   IMPROVED-MAISONETTE = 18
# #   PREMIUM APARTMENT LOFT = 19
# #   PREMIUM MAISONETTE = 20
# 
# =============================================================================

data = dataset_house_type_vs_price.iloc[:,2].values
data_ROOMTYPE_1 = [1 if element == 'IMPROVED' else element for element in data]
data = data_ROOMTYPE_1
data_ROOMTYPE_2 = [2 if element == 'NEW GENERATION' else element for element in data]
data = data_ROOMTYPE_2
data_ROOMTYPE_3 = [3 if element == 'MODEL A' else element for element in data]
data = data_ROOMTYPE_3
data_ROOMTYPE_4 = [4 if element == 'MODEL A2' else element for element in data]
data = data_ROOMTYPE_4
data_ROOMTYPE_5 = [5 if element == 'DBSS' else element for element in data]
data = data_ROOMTYPE_5
data_ROOMTYPE_6 = [6 if element == 'STANDARD' else element for element in data]
data = data_ROOMTYPE_6
data_ROOMTYPE_7 = [7 if element == 'SIMPLIFIED' else element for element in data]
data = data_ROOMTYPE_7
data_ROOMTYPE_8 = [8 if element == 'MAISONETTE' else element for element in data]
data = data_ROOMTYPE_8
data_ROOMTYPE_9 = [9 if element == 'MODEL A-MAISONETTE' else element for element in data]
data = data_ROOMTYPE_9
data_ROOMTYPE_10 = [10 if element == 'APARTMENT' else element for element in data]
data = data_ROOMTYPE_10
data_ROOMTYPE_11 = [11 if element == 'PREMIUM APARTMENT' else element for element in data]
data = data_ROOMTYPE_11
data_ROOMTYPE_12 = [12 if element == 'TYPE S1' else element for element in data]
data = data_ROOMTYPE_12
data_ROOMTYPE_13 = [13 if element == 'TYPE S2' else element for element in data]
data = data_ROOMTYPE_13
data_ROOMTYPE_14 = [14 if element == 'ADJOINED FLAT' else element for element in data]
data = data_ROOMTYPE_14
data_ROOMTYPE_15 = [15 if element == 'MULTI-GENERATION' else element for element in data]
data = data_ROOMTYPE_15
data_ROOMTYPE_16 = [16 if element == 'TERRACE' else element for element in data]
data = data_ROOMTYPE_16
data_ROOMTYPE_17 = [17 if element == '2-ROOM' else element for element in data]
data = data_ROOMTYPE_17
data_ROOMTYPE_18 = [18 if element == 'IMPROVED-MAISONETTE' else element for element in data]
data = data_ROOMTYPE_18
data_ROOMTYPE_19 = [19 if element == 'PREMIUM APARTMENT LOFT' else element for element in data]
data = data_ROOMTYPE_19
data_ROOMTYPE_20 = [20 if element == 'PREMIUM MAISONETTE' else element for element in data]
data = data_ROOMTYPE_20



dataset_house_type_vs_price.iloc[:,2] = data

# =============================================================================
# CONVERTING STOREY RANGE TO INTEGER USING THE AVERAGE
# =============================================================================

data = dataset_house_type_vs_price.iloc[:,3]
data_STOREY_1 = [2 if element == '01 TO 03' else element for element in data]
data = data_STOREY_1
data_STOREY_2 = [5 if element == '04 TO 06' else element for element in data]
data = data_STOREY_2
data_STOREY_3 = [8 if element == '07 TO 09' else element for element in data]
data = data_STOREY_3
data_STOREY_4 = [11 if element == '10 TO 12' else element for element in data]
data = data_STOREY_4
data_STOREY_5 = [14 if element == '13 TO 15' else element for element in data]
data = data_STOREY_5
data_STOREY_6 = [17 if element == '16 TO 18' else element for element in data]
data = data_STOREY_6
data_STOREY_7 = [20 if element == '19 TO 21' else element for element in data]
data = data_STOREY_7
data_STOREY_8 = [23 if element == '22 TO 24' else element for element in data]
data = data_STOREY_8
#01-05
data_STOREY_9 = [3 if element == '01 TO 05' else element for element in data]
data = data_STOREY_9
#06-10
data_STOREY_10 = [8 if element == '06 TO 10' else element for element in data]
data = data_STOREY_10
#11-15
data_STOREY_11 = [13 if element == '11 TO 15' else element for element in data]
data = data_STOREY_11
#16-20
data_STOREY_12 = [18 if element == '16 TO 20' else element for element in data]
data = data_STOREY_12
#21-25
data_STOREY_13 = [23 if element == '21 TO 25' else element for element in data]
data = data_STOREY_13
#25-27
data_STOREY_14 = [26 if element == '25 TO 27' else element for element in data]
data = data_STOREY_14
#26-30
data_STOREY_15 = [28 if element == '26 TO 30' else element for element in data]
data = data_STOREY_15
#31-35
data_STOREY_16 = [33 if element == '31 TO 35' else element for element in data]
data = data_STOREY_16
#28-30
data_STOREY_17 = [29 if element == '28 TO 30' else element for element in data]
data = data_STOREY_17
#31-33
data_STOREY_18 = [32 if element == '31 TO 33' else element for element in data]
data = data_STOREY_18
#34-36
data_STOREY_19 = [35 if element == '34 TO 36' else element for element in data]
data = data_STOREY_19
#37-39
data_STOREY_20 = [38 if element == '37 TO 39' else element for element in data]
data = data_STOREY_20
#36-40
data_STOREY_21 = [38 if element == '36 TO 40' else element for element in data]
data = data_STOREY_21
#40-42
data_STOREY_22 = [41 if element == '40 TO 42' else element for element in data]
data = data_STOREY_22
#43-45
data_STOREY_23 = [44 if element == '43 TO 45' else element for element in data]
data = data_STOREY_23
#46-48
data_STOREY_24 = [47 if element == '46 TO 48' else element for element in data]
data = data_STOREY_24
#49-51
data_STOREY_25 = [50 if element == '49 TO 51' else element for element in data]
data = data_STOREY_25
#
dataset_house_type_vs_price.iloc[:,3] = data

# =============================================================================
# Getting the estimated age of the house from lease commence date
# =============================================================================


feature_columns_to_use3 = ['lease_commence_date','resale_price']

dataset_house_age_vs_price = dataset[feature_columns_to_use3].append(dataset2[feature_columns_to_use3])
dataset_house_age_vs_price = dataset_house_age_vs_price.append(dataset3[feature_columns_to_use3])
dataset_house_age_vs_price = dataset_house_age_vs_price.append(dataset4[feature_columns_to_use3])

age = dataset_house_age_vs_price.iloc[:, 0:1].values
house_age = 2018-age

house_age = pd.DataFrame(house_age)

# =============================================================================
# Getting a unique address list    
# =============================================================================
    
street_name_data_column = ['street_name']

dataset_street_vs_price = dataset[street_name_data_column].append(dataset2[street_name_data_column])
dataset_street_vs_price = dataset_street_vs_price.append(dataset3[street_name_data_column])
dataset_street_vs_price = dataset_street_vs_price.append(dataset4[street_name_data_column])

street_name_data_column = dataset_street_vs_price.drop_duplicates()


# =============================================================================
# GETTING DISTANCE FROM ADDRESS TO WAYPOINT
# =============================================================================


from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from geopy.extra.rate_limiter import RateLimiter

street_name = pd.DataFrame(street_name_data_column.iloc[:,0])
street_name_data_column.to_csv(r"C:\Users\bubu\Downloads\holmusk\streets.csv", index=False)
street_name_dataset = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\streets.csv')

street_lat_long_dataset = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\street_lat_long.csv')



geolocator = Nominatim(user_agent="Address_locator")

#geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

# A recursive function to avoid any possible exceptions
def do_geocode(location):
    try:
# Using orchard central as the centre point of all distance calculation; can  change to other points to augment the data
        return geolocator.geocode(location)
        print((location.latitude, location.longitude))
    except GeocoderTimedOut:
        return do_geocode(location)

# centre point function use
do_geocode("176 Orchard Rd, Singapore 238843")

# centre point current 
location = [1.3, 103.8]


# =============================================================================
# Address latitude and longitude:

x = np.arange(0,547,1)
address_lat = np.array(street_lat_long_dataset.iloc[:,1])
address_long = np.array(street_lat_long_dataset.iloc[:,0])

for i in x-1:
     
     origin = do_geocode(street_name_dataset.iloc[54,0])  
     address_lat = np.append(address_lat, (1.38))    
     address_long = np.append(address_long, (103.9))    

address_long[98] = 103.8

address_lat = address_lat[0:546]
address_location = pd.DataFrame(address_lat, address_long)
address_location.to_csv(r'C:\Users\bubu\Downloads\holmusk\street_lat_long.csv', index = True)


street_data = pd.read_csv(r'C:\Users\bubu\Downloads\holmusk\street_lat_long.csv')



landmark = location

# approximate distance between address and landmark
address = (street_data.iloc[:,1], street_data.iloc[:,0]) # (lat, long)
address = np.array(address)
address = np.transpose(address)

x = np.arange(0,547,1)
distance_approx = []
for i in x-1:
    origin_addr = (address[i,0], address[i,1])

    distance = gpd.vincenty(origin_addr, landmark).km
    distance_approx = np.append(distance_approx, distance)
    
distance_dataset = pd.DataFrame(distance_approx)





# =============================================================================
# # exporting the above dataset into a csv file
# ============================================================================
    
distance_dataset.to_csv(r'C:\Users\bubu\Downloads\holmusk\distance.csv', index=False)
dataset_house_size_vs_price.to_csv("dataset_house_size_vs_price.csv", index=True)
dataset_house_type_vs_price.to_csv(r'C:\Users\bubu\Downloads\holmusk\dataset_house_type_vs_price.csv', index=False)
house_age.to_csv(r'C:\Users\bubu\Downloads\holmusk\house_age.csv', index=False)
