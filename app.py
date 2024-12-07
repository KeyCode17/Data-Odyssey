# Streamlit Core
import streamlit as st
import streamlit_nested_layout
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

# Libraries
import os
import sys


dirloc = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirloc, 'Data Odyssey 1'))
sys.path.append(os.path.join(dirloc, 'Data Odyssey 2'))

from data_odyssey_1 import data_odyssey_1
from data_odyssey_2 import data_odyssey_2

main_menu = option_menu(None, ["Data Odyssey 1", "Data Odyssey 2"], 
    icons=['bi-wifi',  'bi-newspaper'], 
    menu_icon="cast", default_index=0, orientation="horizontal",
    styles={"nav-link": {"font-size": "25px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"}})

st.query_params["Page"] = main_menu

if st.query_params["Page"] == 'Data Odyssey 1':
    data_odyssey_1()

if st.query_params["Page"] == 'Data Odyssey 2':
    data_odyssey_2()
