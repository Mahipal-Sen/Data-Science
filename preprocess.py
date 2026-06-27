"""
=============================================================================
DATA PREPROCESSING & FEATURE ENGINEERING MODULE
=============================================================================
Role: Handles all data preprocessing and feature engineering
Purpose:
  - Standardizes column names to a common format
  - Detects and handles missing values
  - Creates distance features between cities using Haversine formula
  - Handles temporal features (dates, times, day of week)
  - Encodes categorical variables (airlines, cities, stops)
  - Creates polynomial features for better model performance
  - Builds complete preprocessing pipeline for training and inference

Key Features:
  - City Coordinates: 100+ Indian cities with GPS coordinates
  - Distance Calculation: Haversine formula for geographic distance
  - Feature Engineering: Days left, trend factors, seasonal encoding
  - Automatic Column Mapping: Finds price, date, airline columns automatically
  - Validation: Ensures dataset has required columns for training

Usage:
  - Called by train.py for training data
  - Called by app.py for prediction data
=============================================================================
"""

import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectKBest, f_regression

logger = logging.getLogger(__name__)

# City coordinates for distance calculation (latitude, longitude)
CITY_COORDINATES = {
    # Major Indian cities
    'Delhi': (28.6139, 77.2090),
    'New Delhi': (28.6139, 77.2090),
    'Mumbai': (19.0760, 72.8777),
    'Bombay': (19.0760, 72.8777),
    'Bangalore': (12.9716, 77.5946),
    'Banglore': (12.9716, 77.5946),
    'Bengaluru': (12.9716, 77.5946),
    'Chennai': (13.0827, 80.2707),
    'Madras': (13.0827, 80.2707),
    'Kolkata': (22.5726, 88.3639),
    'Goa': (15.2993, 73.8243),
    'Indore': (22.7196, 75.8577),
    'Calcutta': (22.5726, 88.3639),
    'Hyderabad': (17.3850, 78.4867),
    'Pune': (18.5204, 73.8567),
    'Ahmedabad': (23.0225, 72.5714),
    'Jaipur': (26.9124, 75.7873),
    'Surat': (21.1702, 72.8311),
    'Lucknow': (26.8467, 80.9462),
    'Kanpur': (26.4499, 80.3319),
    'Nagpur': (21.1458, 79.0882),
    'Visakhapatnam': (17.6868, 83.2185),
    'Bhopal': (23.2599, 77.4126),
    'Patna': (25.5941, 85.1376),
    'Ludhiana': (30.9010, 75.8573),
    'Agra': (27.1767, 78.0081),
    'Nashik': (19.9975, 73.7898),
    'Faridabad': (28.4089, 77.3178),
    'Meerut': (28.9845, 77.7064),
    'Rajkot': (22.3039, 70.8022),
    'Varanasi': (25.3176, 82.9739),
    'Srinagar': (34.0837, 74.7973),
    'Aurangabad': (19.8762, 75.3433),
    'Dhanbad': (23.7957, 86.4304),
    'Amritsar': (31.6340, 74.8723),
    'Navi Mumbai': (19.0330, 73.0297),
    'Allahabad': (25.4358, 81.8463),
    'Prayagraj': (25.4358, 81.8463),
    'Ranchi': (23.3441, 85.3096),
    'Howrah': (22.5958, 88.2636),
    'Jabalpur': (23.1815, 79.9864),
    'Gwalior': (26.2183, 78.1828),
    'Vijayawada': (16.5062, 80.6480),
    'Jodhpur': (26.2389, 73.0243),
    'Raipur': (21.2514, 81.6296),
    'Kota': (25.2138, 75.8648),
    'Guwahati': (26.1445, 91.7362),
    'Chandigarh': (30.7333, 76.7794),
    'Solapur': (17.6599, 75.9064),
    'Hubballi': (15.3647, 75.1240),
    'Dharwad': (15.3647, 75.1240),
    'Bareilly': (28.3670, 79.4304),
    'Moradabad': (28.8386, 78.7733),
    'Mysore': (12.2958, 76.6394),
    'Gurgaon': (28.4595, 77.0266),
    'Aligarh': (27.8974, 78.0880),
    'Jalandhar': (31.3260, 75.5762),
    'Tiruchirappalli': (10.7905, 78.7047),
    'Bhubaneswar': (20.2961, 85.8245),
    'Salem': (11.6643, 78.1460),
    'Warangal': (17.9784, 79.6000),
    'Guntur': (16.3067, 80.4365),
    'Bhiwandi': (19.2813, 73.0483),
    'Saharanpur': (29.9679, 77.5452),
    'Gorakhpur': (26.7606, 83.3732),
    'Bikaner': (28.0229, 73.3119),
    'Amravati': (20.9374, 77.7796),
    'Noida': (28.5355, 77.3910),
    'Jamshedpur': (22.8046, 86.2029),
    'Bhilai': (21.1938, 81.3509),
    'Cuttack': (20.4625, 85.8830),
    'Firozabad': (27.1591, 78.3958),
    'Kochi': (9.9312, 76.2673),
    'Cochin': (9.9312, 76.2673),
    'Nellore': (14.4426, 79.9865),
    'Bhavnagar': (21.7645, 72.1519),
    'Dehradun': (30.3165, 78.0322),
    'Durgapur': (23.5204, 87.3119),
    'Asansol': (23.6739, 86.9524),
    'Rourkela': (22.2604, 84.8536),
    'Nanded': (19.1383, 77.3210),
    'Kolhapur': (16.7050, 74.2433),
    'Ajmer': (26.4499, 74.6399),
    'Akola': (20.7000, 77.0082),
    'Gulbarga': (17.3297, 76.8343),
    'Jamnagar': (22.4707, 70.0577),
    'Ujjain': (23.1765, 75.7885),
    'Loni': (28.7514, 77.2880),
    'Siliguri': (26.7271, 88.3953),
    'Jhansi': (25.4484, 78.5685),
    'Ulhasnagar': (19.2215, 73.1645),
    'Jammu': (32.7266, 74.8570),
    'Sangli': (16.8544, 74.5642),
    'Mangalore': (12.9141, 74.8560),
    'Erode': (11.3410, 77.7172),
    'Belgaum': (15.8497, 74.4977),
    'Ambattur': (13.1143, 80.1548),
    'Tirunelveli': (8.7139, 77.7567),
    'Malegaon': (20.5537, 74.5288),
    'Gaya': (24.7914, 84.9994),
    'Tiruppur': (11.1085, 77.3411),
    'Davanagere': (14.4644, 75.9218),
    'Kozhikode': (11.2588, 75.7804),
    'Calicut': (11.2588, 75.7804),
    'Akbarpur': (26.4407, 82.5533),
    'Tumkur': (13.3379, 77.1173),
    'Kurnool': (15.8281, 78.0373),
    'Udaipur': (24.5854, 73.7125),
    'Bihar Sharif': (25.1982, 85.5149),
    'South Dumdum': (22.6100, 88.4000),
    'Bihar': (25.0961, 85.3131),
    'Arrah': (25.5560, 84.6643),
    'Panihati': (22.6900, 88.3700),
    'Purnia': (25.7771, 87.4753),
    'Begusarai': (25.4185, 86.1279),
    'Katihar': (25.5427, 87.5714),
    'Sasaram': (24.9490, 84.0287),
    'Hajipur': (25.6854, 85.2098),
    'Dehri': (24.9275, 84.1901),
    'Siwan': (26.2207, 84.3567),
    'Motihari': (26.6469, 84.9181),
    'Nawada': (24.8867, 85.5435),
    'Bagaha': (27.0992, 84.0900),
    'Bettiah': (26.8023, 84.5030),
    'Supaul': (26.1150, 86.5953),
    'Buxar': (25.5647, 83.9777),
    'Jehanabad': (25.2130, 84.9871),
    'Aurangabad': (24.7520, 84.3742),
    'Chapra': (25.7803, 84.7474),
    'Gopalganj': (26.4645, 84.4380),
    'Sitamarhi': (26.5936, 85.4909),
    'Jamui': (24.9170, 86.2247),
    'Munger': (25.3748, 86.4735),
    'Saharsa': (25.8835, 86.6006),
    'Madhepura': (25.9213, 86.7927),
    'Kishanganj': (26.1025, 87.9384),
    'Khagaria': (25.5022, 86.4674),
    'Banka': (24.8809, 86.9226),
    'Lakhisarai': (25.1674, 86.0985),
    'Sheikhpura': (25.1416, 85.8628),
    'Samastipur': (25.8626, 85.7819),
    'Vaishali': (25.7400, 85.2800),
    'Darbhanga': (26.1542, 85.8918),
    'Muzaffarpur': (26.1209, 85.3647),
    'Madhubani': (26.3537, 86.0719),
    'East Champaran': (26.5000, 84.9000),
    'West Champaran': (27.1500, 84.4000),
    'Sitamarhi': (26.5936, 85.4909),
    'Sheohar': (26.5139, 85.2934),
    'Saran': (25.9167, 84.7833),
    'Siwan': (26.2207, 84.3567),
    'Gopalganj': (26.4645, 84.4380),
    'West Champaran': (27.1500, 84.4000),
    'East Champaran': (26.5000, 84.9000),
    'Madhubani': (26.3537, 86.0719),
    'Samastipur': (25.8626, 85.7819),
    'Darbhanga': (26.1542, 85.8918),
    'Muzaffarpur': (26.1209, 85.3647),
    'Vaishali': (25.7400, 85.2800),
    'Sitamarhi': (26.5936, 85.4909),
    'Sheohar': (26.5139, 85.2934),
    'Saran': (25.9167, 84.7833),
    'Chapra': (25.7803, 84.7474),
    'Hajipur': (25.6854, 85.2098),
    'Patna': (25.5941, 85.1376),
    'Nawada': (24.8867, 85.5435),
    'Jehanabad': (25.2130, 84.9871),
    'Aurangabad': (24.7520, 84.3742),
    'Arrah': (25.5560, 84.6643),
    'Buxar': (25.5647, 83.9777),
    'Sasaram': (24.9490, 84.0287),
    'Dehri': (24.9275, 84.1901),
    'Bihar Sharif': (25.1982, 85.5149),
    'Ara': (25.5560, 84.6643),
    'Begusarai': (25.4185, 86.1279),
    'Katihar': (25.5427, 87.5714),
    'Purnia': (25.7771, 87.4753),
    'Saharsa': (25.8835, 86.6006),
    'Madhepura': (25.9213, 86.7927),
    'Supaul': (26.1150, 86.5953),
    'Araria': (26.1325, 87.4559),
    'Kishanganj': (26.1025, 87.9384),
    'Khagaria': (25.5022, 86.4674),
    'Bhagalpur': (25.2370, 87.0070),
    'Banka': (24.8809, 86.9226),
    'Munger': (25.3748, 86.4735),
    'Lakhisarai': (25.1674, 86.0985),
    'Sheikhpura': (25.1416, 85.8628),
    'Jamui': (24.9170, 86.2247),
    'Nalanda': (25.2000, 85.4500),
    'Rohtas': (24.8833, 84.1833),
    'Kaimur': (25.0500, 83.6167),
    'Bhojpur': (25.4667, 84.5167),
    'Buxar': (25.5647, 83.9777),
    'Gaya': (24.7914, 84.9994),
    'Jehanabad': (25.2130, 84.9871),
    'Arwal': (25.2333, 84.6667),
    'Nawada': (24.8867, 85.5435),
    'Aurangabad': (24.7520, 84.3742),
    'Jamui': (24.9170, 86.2247),
    'Banka': (24.8809, 86.9226),
    'Munger': (25.3748, 86.4735),
    'Lakhisarai': (25.1674, 86.0985),
    'Sheikhpura': (25.1416, 85.8628),
    'Khagaria': (25.5022, 86.4674),
    'Bhagalpur': (25.2370, 87.0070),
    'Katihar': (25.5427, 87.5714),
    'Purnia': (25.7771, 87.4753),
    'Araria': (26.1325, 87.4559),
    'Kishanganj': (26.1025, 87.9384),
    'Udalguri': (26.7537, 92.1021),
    'Darrang': (26.4500, 92.0167),
    'Sonitpur': (26.6833, 92.8500),
    'Kokrajhar': (26.4000, 90.2667),
    'Dhubri': (26.0200, 89.9667),
    'Goalpara': (26.1667, 90.6167),
    'Barpeta': (26.3167, 91.0000),
    'Morigaon': (26.2500, 92.3333),
    'Nagaon': (26.3500, 92.6833),
    'Golaghat': (26.5167, 93.9667),
    'Jorhat': (26.7500, 94.2167),
    'Sivasagar': (26.9833, 94.6333),
    'Dibrugarh': (27.4833, 94.9000),
    'Tinsukia': (27.4833, 95.3500),
    'Lakhimpur': (27.2333, 94.1000),
    'Dhemaji': (27.4833, 94.5667),
    'Karbi Anglong': (25.8667, 93.4333),
    'North Cachar Hills': (25.2833, 93.0167),
    'Cachar': (24.7833, 92.8000),
    'Karimganj': (24.8667, 92.3500),
    'Hailakandi': (24.6833, 92.5667),
    'Bongaigaon': (26.4667, 90.5333),
    'Chirang': (26.5167, 90.4833),
    'Kamrup': (26.3167, 91.6833),
    'Kamrup Metropolitan': (26.1500, 91.7667),
    'Nalbari': (26.4500, 91.4333),
    'Baksa': (26.7000, 91.6000),
    'Darrang': (26.4500, 92.0167),
    'Udalguri': (26.7537, 92.1021),
    'Chirang': (26.5167, 90.4833),
    'Bongaigaon': (26.4667, 90.5333),
    'Goalpara': (26.1667, 90.6167),
    'Barpeta': (26.3167, 91.0000),
    'Kamrup': (26.3167, 91.6833),
    'Kamrup Metropolitan': (26.1500, 91.7667),
    'Nalbari': (26.4500, 91.4333),
    'Baksa': (26.7000, 91.6000),
    'Morigaon': (26.2500, 92.3333),
    'Nagaon': (26.3500, 92.6833),
    'Sonitpur': (26.6833, 92.8500),
    'Lakhimpur': (27.2333, 94.1000),
    'Dhemaji': (27.4833, 94.5667),
    'Golaghat': (26.5167, 93.9667),
    'Jorhat': (26.7500, 94.2167),
    'Sivasagar': (26.9833, 94.6333),
    'Dibrugarh': (27.4833, 94.9000),
    'Tinsukia': (27.4833, 95.3500),
    'Karbi Anglong': (25.8667, 93.4333),
    'North Cachar Hills': (25.2833, 93.0167),
    'Cachar': (24.7833, 92.8000),
    'Karimganj': (24.8667, 92.3500),
    'Hailakandi': (24.6833, 92.5667),
    'Dimapur': (25.9063, 93.7276),
    'Kohima': (25.6747, 94.1100),
    'Wokha': (26.1000, 94.2667),
    'Mokokchung': (26.3167, 94.5167),
    'Tuensang': (26.2833, 94.8167),
    'Mon': (26.7167, 95.0333),
    'Phek': (25.6667, 94.5000),
    'Zunheboto': (25.9667, 94.5167),
    'Kiphire': (25.3500, 94.4833),
    'Longleng': (26.4833, 94.8167),
    'Peren': (25.5167, 93.7333),
    'Imphal': (24.8170, 93.9368),
    'Thoubal': (24.6333, 94.0167),
    'Bishnupur': (24.6333, 93.7667),
    'Churachandpur': (24.3333, 93.6833),
    'Chandel': (24.3167, 94.0167),
    'Senapati': (25.2833, 94.0167),
    'Tamenglong': (24.9833, 93.4833),
    'Ukhrul': (25.1000, 94.3667),
    'East Imphal': (24.7833, 93.9667),
    'West Imphal': (24.8167, 93.9333),
    'Aizawl': (23.7271, 92.7176),
    'Lunglei': (22.8833, 92.7333),
    'Saiha': (22.4833, 92.9667),
    'Champhai': (23.4667, 93.3167),
    'Kolasib': (24.2333, 92.6833),
    'Mamit': (23.9333, 92.4833),
    'Serchhip': (23.3167, 92.8333),
    'Lawngtlai': (22.5333, 92.8833),
    'Saitual': (23.9667, 92.5667),
    'Khawzawl': (23.3000, 93.0167),
    'Hnahthial': (22.9667, 92.8833),
    'Shillong': (25.5788, 91.8933),
    'East Khasi Hills': (25.5500, 91.8833),
    'West Khasi Hills': (25.3167, 91.2833),
    'Ri Bhoi': (25.9000, 91.8667),
    'East Garo Hills': (25.6667, 90.6167),
    'West Garo Hills': (25.5667, 90.2167),
    'South Garo Hills': (25.2833, 90.2833),
    'North Garo Hills': (25.9000, 90.6167),
    'Jaintia Hills': (25.3167, 92.1333),
    'East Jaintia Hills': (25.3167, 92.1333),
    'West Jaintia Hills': (25.1167, 92.2500),
    'Agartala': (23.8315, 91.2868),
    'West Tripura': (23.9167, 91.2833),
    'South Tripura': (23.2500, 91.2833),
    'Dhalai': (23.8500, 91.9167),
    'North Tripura': (24.0833, 92.0000),
    'Khowai': (24.0833, 91.6167),
    'Sepahijala': (23.7500, 91.2833),
    'Gomati': (23.4833, 91.6167),
    'Unakoti': (24.2833, 92.0000),
    'Sipahijala': (23.7500, 91.2833),
    'Gangtok': (27.3389, 88.6065),
    'East Sikkim': (27.3333, 88.6167),
    'West Sikkim': (27.2833, 88.3667),
    'North Sikkim': (27.6167, 88.4833),
    'South Sikkim': (27.2833, 88.6167),
    'Itanagar': (27.0844, 93.6053),
    'Tawang': (27.5833, 91.8667),
    'West Kameng': (27.2833, 92.2833),
    'East Kameng': (27.4167, 93.0167),
    'Papum Pare': (27.0833, 93.6167),
    'Kurung Kumey': (27.9167, 93.4167),
    'Kra Daadi': (28.1667, 93.6167),
    'Lower Subansiri': (27.5167, 93.8167),
    'Upper Subansiri': (27.7667, 94.0833),
    'West Siang': (28.1667, 94.7667),
    'East Siang': (28.0833, 95.2833),
    'Siang': (28.0833, 95.2833),
    'Upper Siang': (28.7667, 95.2833),
    'Lower Siang': (27.4833, 94.6167),
    'Lepa Rada': (28.3167, 95.0167),
    'Lower Dibang Valley': (27.4833, 95.6167),
    'Dibang Valley': (28.0833, 95.6167),
    'Anjaw': (28.0667, 96.6167),
    'Lohit': (27.8833, 96.2833),
    'Namsai': (27.6833, 95.8667),
    'Changlang': (27.2833, 95.6167),
    'Tirap': (27.0167, 95.5333),
    'Longding': (26.8667, 95.3167),
    'Pakke Kessang': (27.0833, 93.2833),
    'Kamle': (27.4000, 93.4500),
    'Shi Yomi': (28.6667, 97.0333),
    'Port Blair': (11.6234, 92.7265),
    'South Andaman': (11.6667, 92.7333),
    'Nicobar': (7.0000, 93.5000),
    'North and Middle Andaman': (12.6167, 92.9500),
    'Car Nicobar': (9.1667, 92.7667),
    'Chandigarh': (30.7333, 76.7794),
    'Daman': (20.3974, 72.8328),
    'Diu': (20.7141, 70.9873),
    'Dadra and Nagar Haveli': (20.1809, 73.0169),
    'Puducherry': (11.9416, 79.8083),
    'Karaikal': (10.9254, 79.8380),
    'Mahe': (11.7022, 75.5347),
    'Yanam': (16.7331, 82.2167),
    'Lakshadweep': (10.5667, 72.6167),
    'Kavaratti': (10.5667, 72.6167),
    'Minicoy': (8.2833, 73.0500),
    'Agatti': (10.8500, 72.1833),
    'Amini': (11.1167, 72.7333),
    'Androth': (10.8167, 73.6833),
    'Bitra': (11.5500, 72.1667),
    'Chetlat': (11.7000, 75.5333),
    'Kadmat': (11.2333, 72.7833),
    'Kalpeni': (10.0667, 73.6500),
    'Kiltan': (11.4833, 73.0167),
    'Pitti': (11.4500, 72.1333),
}

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on the earth."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of earth in kilometers
    return c * r

def calculate_city_distance(source: str, destination: str) -> float:
    """Calculate distance between two cities using their coordinates."""
    source = source.strip().title() if source else ""
    destination = destination.strip().title() if destination else ""

    # Handle common variations
    source_coords = None
    dest_coords = None

    # Try exact match first
    if source in CITY_COORDINATES:
        source_coords = CITY_COORDINATES[source]
    if destination in CITY_COORDINATES:
        dest_coords = CITY_COORDINATES[destination]

    # If not found, try some common variations
    if not source_coords:
        if 'Delhi' in source:
            source_coords = CITY_COORDINATES.get('Delhi')
        elif 'Mumbai' in source or 'Bombay' in source:
            source_coords = CITY_COORDINATES.get('Mumbai')
        elif 'Bangalore' in source or 'Banglore' in source:
            source_coords = CITY_COORDINATES.get('Bangalore')
        elif 'Chennai' in source or 'Madras' in source:
            source_coords = CITY_COORDINATES.get('Chennai')
        elif 'Kolkata' in source or 'Calcutta' in source:
            source_coords = CITY_COORDINATES.get('Kolkata')
        elif 'Hyderabad' in source:
            source_coords = CITY_COORDINATES.get('Hyderabad')
        elif 'Pune' in source:
            source_coords = CITY_COORDINATES.get('Pune')
        elif 'Kochi' in source or 'Cochin' in source:
            source_coords = CITY_COORDINATES.get('Kochi')

    if not dest_coords:
        if 'Delhi' in destination:
            dest_coords = CITY_COORDINATES.get('Delhi')
        elif 'Mumbai' in destination or 'Bombay' in destination:
            dest_coords = CITY_COORDINATES.get('Mumbai')
        elif 'Bangalore' in destination or 'Banglore' in destination:
            dest_coords = CITY_COORDINATES.get('Bangalore')
        elif 'Chennai' in destination or 'Madras' in destination:
            dest_coords = CITY_COORDINATES.get('Chennai')
        elif 'Kolkata' in destination or 'Calcutta' in destination:
            dest_coords = CITY_COORDINATES.get('Kolkata')
        elif 'Hyderabad' in destination:
            dest_coords = CITY_COORDINATES.get('Hyderabad')
        elif 'Pune' in destination:
            dest_coords = CITY_COORDINATES.get('Pune')
        elif 'Kochi' in destination or 'Cochin' in destination:
            dest_coords = CITY_COORDINATES.get('Kochi')

    if source_coords and dest_coords:
        return haversine_distance(source_coords[0], source_coords[1], dest_coords[0], dest_coords[1])
    else:
        # Return a default distance if cities not found
        logger.warning(f"Could not find coordinates for {source} or {destination}, using default distance")
        return 500.0  # Default 500km

COLUMN_ALIASES = {
    'Airline': ['Airline', 'Carrier', 'Airline_Name', 'airline', 'Carrier_name'],
    'Source': ['Source', 'From', 'Origin', 'Departure_City', 'from', 'Departure'],
    'Destination': ['Destination', 'To', 'Dest', 'Arrival_City', 'to', 'Arrival'],
    'Date_of_Journey': ['Date_of_Journey', 'Date', 'Journey_Date', 'Travel_Date', 'flight_date', 'journey_date', 'Date_of_Travel'],
    'Dep_Time': ['Dep_Time', 'Departure_Time', 'Departure', 'DepTime', 'dep_time', 'departure_time'],
    'Arrival_Time': ['Arrival_Time', 'Arrival', 'ArrTime', 'arr_time', 'arrival_time'],
    'Duration': ['Duration', 'Flight_Duration', 'Travel_Time', 'time_taken', 'duration', 'Flight time'],
    'Total_Stops': ['Total_Stops', 'Stops', 'Stop_Count', 'stop', 'Number_of_Stops', 'Total_Stops'],
    'Route': ['Route', 'path', 'travel_route'],
    'Additional_Info': ['Additional_Info', 'Info', 'AdditionalInfo', 'Notes', 'Note']
}
TARGET_ALIASES = ['price', 'fare', 'ticket', 'cost', 'amount', 'rate']

DEFAULT_REQUIRED_FEATURES = [
    'Airline',
    'Source',
    'Destination',
    'Date_of_Journey',
    'Dep_Time',
    'Duration'
]

REQUIRED_ROLE_MAP = {
    'Airline': 'airline',
    'Source': 'source',
    'Destination': 'destination',
    'Date_of_Journey': 'date',
    'Dep_Time': 'dep_time',
    'Duration': 'duration'
}

OPTIONAL_FEATURES = ['Arrival_Time', 'Total_Stops', 'Route', 'Additional_Info']


def _clean_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract date features from date column."""
    
    def __init__(self, date_col_name: str = 'Date_of_Journey'):
        self.date_col_name = date_col_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.date_col_name in X.columns:
            # Parse dates with multiple formats
            dates = self._parse_dates(X[self.date_col_name])
            X['Journey_day'] = dates.dt.day.fillna(0).astype(int)
            X['Journey_month'] = dates.dt.month.fillna(0).astype(int)
            X['Journey_year'] = dates.dt.year.fillna(0).astype(int)
            X['Journey_quarter'] = dates.dt.quarter.fillna(0).astype(int)
            X['Journey_week'] = dates.dt.isocalendar().week.fillna(0).astype(int)
            X['Journey_weekday'] = dates.dt.weekday.fillna(0).astype(int)
            X['Is_Weekend'] = (dates.dt.weekday >= 5).astype(int).fillna(0)
        return X
    
    @staticmethod
    def _parse_dates(series: pd.Series) -> pd.Series:
        """Parse dates in multiple formats."""
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        
        series = series.astype(str).replace(r'^\s*$', np.nan, regex=True)
        formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%m-%d-%Y', '%d.%m.%Y', '%Y/%m/%d']
        
        for fmt in formats:
            parsed = pd.to_datetime(series, format=fmt, errors='coerce')
            if not parsed.isna().all():
                return parsed
        
        parsed = pd.to_datetime(series, dayfirst=True, errors='coerce')
        if parsed.isna().all():
            parsed = pd.to_datetime(series, errors='coerce')
        return parsed


def add_days_left_feature(df: pd.DataFrame, journey_date_col: str = 'Date_of_Journey', current_date: Optional[datetime.date] = None) -> pd.DataFrame:
    """Add a booking window feature representing days left before journey."""
    df = df.copy()
    if current_date is None:
        current_date = datetime.now().date()

    if journey_date_col not in df.columns:
        df['days_left'] = 0
        return df

    dates = DateFeatureExtractor._parse_dates(df[journey_date_col])
    today = pd.to_datetime(current_date)
    days_left = (dates - today).dt.days.fillna(0).astype(int)
    days_left = days_left.clip(lower=0)
    df['days_left'] = days_left
    return df


def add_year_trend(df: pd.DataFrame, journey_date_col: str = 'Date_of_Journey', base_year: int = 2019) -> pd.DataFrame:
    """Add a yearly trend factor to simulate inflation and market growth."""
    df = df.copy()
    if journey_date_col not in df.columns:
        df['trend_factor'] = 1.0
        return df

    dates = DateFeatureExtractor._parse_dates(df[journey_date_col])
    years_passed = (dates.dt.year - base_year).fillna(0).astype(int)
    years_passed = years_passed.clip(lower=0)
    df['trend_factor'] = 1.0 + (0.05 * years_passed)
    return df


class TimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract hour and minute from time columns."""
    
    def __init__(self, dep_time_col: str = 'Dep_Time', arrival_time_col: str = 'Arrival_Time'):
        self.dep_time_col = dep_time_col
        self.arrival_time_col = arrival_time_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Extract departure time features
        if self.dep_time_col in X.columns:
            dep_times = self._parse_time(X[self.dep_time_col])
            X['Dep_hour'] = dep_times['hour']
            X['Dep_minute'] = dep_times['minute']
            X['Is_Peak_Hour'] = X['Dep_hour'].apply(lambda x: 1 if (6 <= x <= 9) or (17 <= x <= 20) else 0)
            X['Is_Morning_Dep'] = X['Dep_hour'].apply(lambda x: 1 if 5 <= x < 12 else 0)
            X['Is_Afternoon_Dep'] = X['Dep_hour'].apply(lambda x: 1 if 12 <= x < 17 else 0)
            X['Is_Evening_Dep'] = X['Dep_hour'].apply(lambda x: 1 if 17 <= x < 22 else 0)
            X['Is_Night_Dep'] = X['Dep_hour'].apply(lambda x: 1 if x >= 22 or x < 5 else 0)
        
        # Extract arrival time features
        if self.arrival_time_col in X.columns:
            arrival_times = self._parse_time(X[self.arrival_time_col])
            X['Arrival_hour'] = arrival_times['hour']
            X['Arrival_minute'] = arrival_times['minute']
        else:
            X['Arrival_hour'] = 0
            X['Arrival_minute'] = 0
        
        return X
    
    @staticmethod
    def _parse_time(series: pd.Series) -> pd.DataFrame:
        """Parse time strings to hour and minute."""
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        
        series = series.astype(str).replace(r'^\s*$', np.nan, regex=True)
        extracted = series.str.extract(r'(?P<time>\d{1,2}:\d{2})')
        parsed = pd.to_datetime(extracted['time'], format='%H:%M', errors='coerce')
        
        return pd.DataFrame({
            'hour': parsed.dt.hour.fillna(0).astype(int),
            'minute': parsed.dt.minute.fillna(0).astype(int)
        })


class DurationConverter(BaseEstimator, TransformerMixin):
    """Convert duration strings to total minutes."""
    
    def __init__(self, duration_col: str = 'Duration'):
        self.duration_col = duration_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.duration_col in X.columns:
            X['Duration_minutes'] = X[self.duration_col].apply(self._convert_duration)
        else:
            X['Duration_minutes'] = 0
        return X
    
    @staticmethod
    def _convert_duration(value: Union[str, float, int, None]) -> int:
        """Convert various duration formats to minutes."""
        if pd.isna(value):
            return 0
        
        text = str(value).strip().lower().replace(' ', '')
        if text == '' or text in ['nan', 'none']:
            return 0
        
        if text.isdigit():
            return int(text)
        
        hours = 0
        minutes = 0
        
        if 'h' in text:
            parts = text.split('h')
            hours = int(parts[0]) if parts[0].isdigit() else 0
            text = parts[1] if len(parts) > 1 else ''
        
        if 'm' in text:
            minutes_part = text.replace('m', '')
            minutes = int(minutes_part) if minutes_part.isdigit() else 0
        elif text.isdigit() and hours == 0:
            minutes = int(text)
        
        return hours * 60 + minutes


class StopsConverter(BaseEstimator, TransformerMixin):
    """Convert stops column to numeric."""
    
    def __init__(self, stops_col: str = 'Total_Stops'):
        self.stops_col = stops_col
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.stops_col in X.columns:
            X['Total_Stops'] = X[self.stops_col].apply(self._convert_stops)
        else:
            X['Total_Stops'] = 0
        return X
    
    @staticmethod
    def _convert_stops(value: Union[str, float, int, None]) -> int:
        """Convert stops to numeric."""
        if pd.isna(value):
            return 0
        text = str(value).strip().lower()
        if text in ['non-stop', 'nonstop', '0', '0 stops', '0 stop']:
            return 0
        
        digits = re.findall(r'\d+', text)
        if digits:
            return int(digits[0])
        return 0


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename_map = {}
    lower_map = {str(col).lower(): col for col in df.columns}

    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            alias_clean = str(alias).lower()
            for raw_col in df.columns:
                raw_col_str = str(raw_col)
                raw_col_lower = raw_col_str.lower()
                if alias_clean == raw_col_lower or alias_clean in raw_col_lower or _clean_name(alias) == _clean_name(raw_col):
                    rename_map[raw_col] = standard_name
                    break
            if standard_name in rename_map.values():
                break

    df = df.rename(columns=rename_map)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df


def detect_target_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [col for col in df.columns if any(token in _clean_name(col) for token in TARGET_ALIASES)]
    numeric_candidates = [col for col in df.select_dtypes(include=['number']).columns if col not in candidates]

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        for token in ['price', 'fare', 'ticket', 'cost']:
            for col in candidates:
                if token in _clean_name(col):
                    return col
        return candidates[0]

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    if candidates:
        return candidates[0]

    return None


def _tokenize_column_name(column_name: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", str(column_name).lower())


def find_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    for keyword in keywords:
        keyword_clean = _clean_name(keyword)
        for col in df.columns:
            col_clean = _clean_name(col)
            if keyword_clean == col_clean:
                return col
            tokens = _tokenize_column_name(col)
            if keyword_clean in tokens:
                return col
    return None


def infer_roles(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    df = standardize_columns(df)
    roles = {
        'target': detect_target_column(df),
        'date': find_column(df, ['journey', 'date', 'travel_date']),
        'dep_time': find_column(df, ['dep_time', 'departure_time', 'departure', 'dep time', 'dep']),
        'arrival_time': find_column(df, ['arrival_time', 'arrival', 'arr_time', 'arr']),
        'duration': find_column(df, ['duration', 'flight_duration', 'travel_time', 'time_taken']),
        'stops': find_column(df, ['total_stops', 'stops', 'stop_count', 'number_of_stops', 'stop']),
        'airline': find_column(df, ['airline', 'carrier']),
        'source': find_column(df, ['source', 'from', 'origin']),
        'destination': find_column(df, ['destination', 'to', 'dest', 'arrival_city']),
        'route': find_column(df, ['route', 'path', 'travel_route']),
        'additional_info': find_column(df, ['additional_info', 'info', 'notes', 'note'])
    }
    return roles


def validate_required_features(roles: Dict[str, Optional[str]]) -> Tuple[bool, List[str]]:
    missing = [feature for feature, role_key in REQUIRED_ROLE_MAP.items() if roles.get(role_key) is None]
    return len(missing) == 0, missing


def normalize_route(route: Union[str, float, int, None]) -> str:
    """Normalize route strings."""
    if pd.isna(route):
        return ''
    text = str(route).strip()
    text = text.replace('→', '->').replace('–', '-').replace('—', '-')
    text = re.sub(r'\s*-\s*>\s*', '->', text)
    text = re.sub(r'\s*->\s*', '->', text)
    text = re.sub(r'\s*-\s*', '->', text)
    return text


def route_segment_count(route: str) -> int:
    """Count segments in a route."""
    if not route or pd.isna(route):
        return 0
    normalized = normalize_route(route)
    parts = [part.strip() for part in re.split(r'->|\u2192', normalized) if part.strip()]
    return max(1, len(parts))


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def compute_arrival_from_duration(dep_series: pd.Series, duration_minutes: pd.Series) -> pd.DataFrame:
    date_series = pd.to_datetime(dep_series, errors='coerce')
    arrival = date_series + pd.to_timedelta(duration_minutes, unit='m')
    return pd.DataFrame({'Arrival_hour': arrival.dt.hour.fillna(0).astype(int), 'Arrival_minute': arrival.dt.minute.fillna(0).astype(int)})


def build_feature_dataframe(df: pd.DataFrame, roles: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, List[str]]]:
    """Build feature dataframe with comprehensive preprocessing."""
    df = df.copy()
    df = standardize_columns(df)
    roles = infer_roles(df)
    df = clean_text_columns(df)

    # Extract target
    if roles['target'] in df.columns:
        target = df[roles['target']].copy()
        if isinstance(target, pd.DataFrame):
            target = target.iloc[:, 0]
    else:
        target = None

    # Apply feature extractors
    extractors = [
        DateFeatureExtractor(roles['date']),
        TimeFeatureExtractor(roles['dep_time'], roles.get('arrival_time')),
        DurationConverter(roles['duration']),
        StopsConverter(roles.get('stops'))
    ]
    
    for extractor in extractors:
        df = extractor.fit_transform(df)

    # Add booking window and yearly trend features
    df = add_days_left_feature(df, roles['date'])
    df = add_year_trend(df, roles['date'])

    # Handle route features
    route_col = roles.get('route')
    if route_col not in df.columns or df[route_col].isna().all():
        df['Route'] = df[roles['source']].astype(str) + ' -> ' + df[roles['destination']].astype(str)
    else:
        df['Route'] = df[route_col].astype(str)

    df['Route'] = df['Route'].apply(normalize_route)
    df['Route_segments'] = df['Route'].apply(route_segment_count)
    df['Route_length'] = df['Route'].apply(lambda x: len(re.split(r'->|\u2192', str(x))))
    df['Route_stops_inferred'] = df['Route_segments'].apply(lambda x: max(0, x - 1))
    df['Has_stops_mismatch'] = (df['Route_stops_inferred'] != df['Total_Stops']).astype(int)

    # Handle additional info
    info_col = roles.get('additional_info')
    if info_col not in df.columns:
        df['Additional_Info'] = 'No info'
    elif df[info_col].isna().all():
        df['Additional_Info'] = 'No info'
    else:
        df['Additional_Info'] = df[info_col].astype(str)

    # Create composite features
    df['Source_Destination'] = df[roles['source']].astype(str) + '_' + df[roles['destination']].astype(str)
    df['Airline_Route'] = df[roles['airline']].astype(str) + '_' + df['Source_Destination']

    # Calculate geographical distance between source and destination
    df['Distance_km'] = df.apply(lambda row: calculate_city_distance(
        str(row[roles['source']]), str(row[roles['destination']])
    ), axis=1)

    # Create stop-related features
    df['Is_one_stop'] = df['Total_Stops'].apply(lambda x: 1 if x == 1 else 0)
    df['Is_nonstop'] = df['Total_Stops'].apply(lambda x: 1 if x == 0 else 0)
    df['Is_multicity'] = df['Total_Stops'].apply(lambda x: 1 if x >= 2 else 0)

    # Define feature columns
    core_features = [roles['airline'], roles['source'], roles['destination'], 'Source_Destination', 'Route', 'Additional_Info']
    numeric_features = [
        'Journey_day', 'Journey_month', 'Journey_year', 'Journey_quarter', 'Journey_week',
        'Dep_hour', 'Dep_minute', 'Duration_minutes', 'Total_Stops', 'Distance_km',
        'Arrival_hour', 'Arrival_minute', 'Journey_weekday', 'Is_Weekend',
        'Is_Peak_Hour', 'Is_Morning_Dep', 'Is_Afternoon_Dep', 'Is_Evening_Dep', 'Is_Night_Dep',
        'Route_segments', 'Route_length', 'Route_stops_inferred', 'Has_stops_mismatch',
        'Is_one_stop', 'Is_nonstop', 'Is_multicity', 'days_left', 'trend_factor'
    ]

    X = df.copy()
    feature_columns = []
    categorical_columns = []

    for col in core_features:
        if col and col in X.columns:
            categorical_columns.append(col)
            feature_columns.append(col)

    for col in numeric_features:
        if col in X.columns:
            feature_columns.append(col)

    X = X[feature_columns].copy()

    # Clean target
    if target is not None:
        cleaned_target = target.astype(str).str.replace(r'[^0-9.-]+', '', regex=True)
        y = pd.to_numeric(cleaned_target, errors='coerce')
    else:
        y = None

    # Clean data
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=0, how='any').reset_index(drop=True)
    if y is not None:
        y = y.loc[X.index].reset_index(drop=True)

    # Categorize features for pipeline
    low_cardinality = []
    high_cardinality = []
    for col in categorical_columns:
        cardinality = X[col].nunique(dropna=False)
        if cardinality <= 10:
            low_cardinality.append(col)
        else:
            high_cardinality.append(col)

    metadata = {
        'feature_columns': feature_columns,
        'numeric_features': [col for col in numeric_features if col in X.columns],
        'categorical_low_cardinality': low_cardinality,
        'categorical_high_cardinality': high_cardinality,
        'roles': roles,
        'target': roles['target']
    }

    return X, y, metadata


def build_training_pipeline(metadata: dict, model) -> Pipeline:
    numeric_cols = metadata['numeric_features']
    low_card = metadata['categorical_low_cardinality']
    high_card = metadata['categorical_high_cardinality']

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    transformers = []
    if numeric_cols:
        transformers.append(('numeric', numeric_pipeline, numeric_cols))

    if low_card:
        low_card_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('low_card', low_card_pipeline, low_card))

    if high_card:
        high_card_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        transformers.append(('high_card', high_card_pipeline, high_card))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


def infer_dataset_profile(df: pd.DataFrame) -> Dict[str, object]:
    df = standardize_columns(df)
    roles = infer_roles(df)
    detected_target = detect_target_column(df)

    profile = {
        'roles': roles,
        'target': detected_target,
        'available_columns': df.columns.tolist(),
        'missing_required': [feature for feature, role_key in REQUIRED_ROLE_MAP.items() if roles.get(role_key) is None]
    }
    return profile


def validate_ready_for_training(df: pd.DataFrame) -> Tuple[bool, str]:
    df = standardize_columns(df)
    roles = infer_roles(df)
    valid, missing = validate_required_features(roles)
    if not valid:
        return False, f"Missing required fields: {', '.join(missing)}"
    if detect_target_column(df) is None:
        return False, 'Target column could not be detected automatically. Please include a Price/Fare field.'
    return True, 'Dataset is ready for training.'


def build_input_dataframe(raw_data: Dict[str, object], roles: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Build input dataframe for prediction from raw user input."""
    df = pd.DataFrame([raw_data])
    df = standardize_columns(df)
    X, _, metadata = build_feature_dataframe(df, roles)
    return X, metadata


def create_prediction_pipeline(metadata: dict, model) -> Pipeline:
    """Create a prediction pipeline that matches the training pipeline."""
    return build_training_pipeline(metadata, model)


def validate_prediction_input(input_data: Dict[str, Any], roles: Dict[str, Optional[str]]) -> Tuple[bool, str]:
    """Validate prediction input data."""
    required_fields = ['airline', 'source', 'destination', 'date', 'dep_time', 'duration']
    missing = [field for field in required_fields if roles.get(field) is None or input_data.get(roles[field]) in [None, '', 'nan']]

    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"

    date_col = roles.get('date')
    if date_col is None:
        return False, 'Journey date column mapping is missing.'

    date_value = input_data.get(date_col)
    if date_value is None or str(date_value).strip() == '':
        return False, 'Journey date is required for prediction.'

    parsed_date = DateFeatureExtractor._parse_dates(pd.Series([date_value]))
    if parsed_date.isna().all():
        return False, 'Journey date is invalid. Provide a valid journey date.'

    return True, "Input is valid."


def preprocess_prediction_data(input_data: Dict[str, Any], roles: Dict[str, Optional[str]]) -> pd.DataFrame:
    """Preprocess single prediction input."""
    # Validate input
    valid, message = validate_prediction_input(input_data, roles)
    if not valid:
        raise ValueError(message)
    
    # Build feature dataframe
    X, metadata = build_input_dataframe(input_data, roles)
    return X


if __name__ == '__main__':
    sample = pd.DataFrame([
        {
            'Airline': 'IndiGo',
            'Source': 'Banglore',
            'Destination': 'New Delhi',
            'Date_of_Journey': '24/03/2019',
            'Dep_Time': '22:20',
            'Arrival_Time': '01:10',
            'Duration': '2h 50m',
            'Total_Stops': 'non-stop',
            'Price': 3897
        },
        {
            'Airline': 'Air India',
            'Source': 'Kolkata',
            'Destination': 'Banglore',
            'Date_of_Journey': '01/05/2019',
            'Dep_Time': '05:50',
            'Arrival_Time': '13:15',
            'Duration': '7h 25m',
            'Total_Stops': '2 stops',
            'Price': 7662
        }
    ])
    print('Profile:', infer_dataset_profile(sample))
    X, y, metadata = build_feature_dataframe(sample, infer_roles(sample))
    print('X shape:', X.shape)
    print('Features:', metadata['feature_columns'])
