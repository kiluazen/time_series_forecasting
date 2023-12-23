import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX 
def main():
    st.title("T-Shirt Inventory Prediction")
    # Dropdown menu for the first part of the SKU
    first_part_options = [""] +['CR-V22-High-Vis-Orange', 'CR-V22-High-Vis-Yellow', 'CR-V32-High-Vis-Orange', 'CR-V32-High-Vis-Yellow']  # Replace with your options
    selected_first_part = st.selectbox("Select the first part of the SKU", first_part_options)

    # Dropdown menu for sizes
    size_options = [""] +[ 'XL','L','M','S','XS','2XL', '3XL', '5XL', '7XL']  # Replace with your size options
    selected_size = st.selectbox("Select the size", size_options)
    if selected_first_part == "" or selected_size == "":
        st.write("Please choose both the First Part and Size of the SKU.")
    else:
        # Streamlit UI elements (input, buttons, etc.)
        tshirt_sku = f'{selected_first_part}-{selected_size}'
        # More input elements as needed...
        # tshirt_sku = 'CR-V22-High-Vis-Orange-L'
        name_list = [ val.lower() for val in tshirt_sku.split('-') ]
        # Get the csv file path where the history of sales is recorded
        file_name = f'{name_list[1]} hv {name_list[-2]} 20211201 20231215.csv'
        path = './data/' + file_name

        data = pd.read_csv(path)
        data['date'] = pd.to_datetime(data['date'], dayfirst= True)
        data.set_index('date', inplace = True)
        df = pd.DataFrame()
        if tshirt_sku not in data.columns:
            st.write(f'Oops don\'t have data for {tshirt_sku} ')
        else:
            df['y'] = data[tshirt_sku]
            # p,q,d ,m params for AIMA algorithm
            od = (0,0,0)
            sod = (2,1,0,12)
            model = SARIMAX( df['y'],
                            order = od, seasonal_order=sod
                            )
            result = model.fit()
            # print(result.summary())
            start = len(df) 
            future_days = 200
            end = len(df) + future_days -1
            predictions = result.predict(start, end, typ = 'levels').rename("Predictions") 
            df_prediction = pd.DataFrame(predictions)

            prediction_array = np.array(df_prediction['Predictions'])

            # Let's get the inventory data
            inventory_data = pd.read_csv('./data/inventory-18_12_2023.csv')
            inventory_sku= inventory_data[inventory_data['SKU'] == tshirt_sku]
            committed = inventory_sku['committed'].values[0]
            totalLocationsQuantity = inventory_sku['totalLocationsQuantity'].values[0]
            net_tobe_sold = totalLocationsQuantity - committed

            assert net_tobe_sold > 0, 'The inventory is already empty'
            index_inventory_end = None
            dp_list = [0]
            for i in range(len(prediction_array)):
                dp_list.append(dp_list[i] + prediction_array[i])
                if dp_list[-1] > net_tobe_sold:
                    index_inventory_end = i
                    break
                
            if not index_inventory_end:
                st.markdown(f'The {tshirt_sku} inventory is good for next {future_days} days, <p>Note: I am ignoring the calculation of exact date if its not under 200 days</p>', unsafe_allow_html=True)
                index_inventory_end = len(prediction_array) -1

            inventory_finish_date = np.array(df_prediction.index)[index_inventory_end]

            # Display results using Streamlit
            st.write(f"No of tshirts yet to be sold: <span style='color:green'>{net_tobe_sold}</span>", unsafe_allow_html=True)
            if index_inventory_end !=future_days-1:
                st.write(f"The inventory for <span style ='color:green'>{tshirt_sku}</span> will end on <span style = 'color:red'>{str(inventory_finish_date)[:10]}</span> which is {index_inventory_end +1} days from 2023-12-15", unsafe_allow_html= True)

if __name__ == "__main__":
    main()
