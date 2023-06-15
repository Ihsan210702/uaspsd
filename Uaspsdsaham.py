import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error 

st.set_page_config(
    page_title="C02 Emission",
    page_icon='https://logowik.com/content/uploads/images/ramayana6876.logowik.com.webp',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""<h1>Peramalan Saham PT Ramayana Lestari Sentosa Tbk </h1>""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h2 style = "text-align: center;"><img src="https://logowik.com/content/uploads/images/ramayana6876.logowik.com.webp" width="130" height="130"><br></h2>""",unsafe_allow_html=True), 
        ["Home", "Description", "Dataset", "Prepocessing", "Modeling", "Implementation"], 
            icons=['house', 'file-earmark-font', 'bar-chart', 'gear', 'arrow-down-square', 'check2-square'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#f8c7b8"},
                "icon": {"color": "black", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"black"},
                "nav-link-selected":{"background-color": "#f8f8d8"}
            }
        )

    if selected == "Home":
        st.write("""<h3 style = "text-align: center;">
        <img src="https://geographical.co.uk/wp-content/uploads/carbon-dioxide-emissions-featured.jpg" width="500" height="300">
        </h3>""",unsafe_allow_html=True)

    elif selected == "Description":
        st.subheader("""Pengertian""")
        st.write("""
        Dataset ini merupakan data saham perusahaan Ramayana Lestari Sentosa jaringan toko swalayan yang memiliki banyak cabang di Indonesia. 
        """)

        st.subheader("""Kegunaan Dataset""")
        st.write("""
        Dataset  ini digunakan untuk melakukan prediksi harga nilai kedepannya untuk
        mengantisipasi kerugian yang didapat oleh pelaku pemain
        saham. Oleh karena itu diperlukan prediksi harga saham untuk
        para pemain saham tidak salah mengambil langkah untuk
        menjual atau membeli saham dalam suatu perusahaan, karena
        data yang diperoleh dari suatu perusahaan berupa data harian
        sehingga akan dapat pergerakan nilai saham secara rinci. 
        """)

        st.subheader("""Fitur""")
        st.markdown(
            """
            Fitur Fitur-fitur yang terdapat pada dataset: Open , High, Low, Close, Adj Close, Volume
            """
        )

        st.subheader("""Sumber Dataset""")
        st.write("""
        Sumber Dataset Sumber data di dapatkan melalui website finance.yahoo.com. Berikut merupakan link untuk mengakses sumber dataset 
        <a href="https://finance.yahoo.com/quote/RALS.JK/history/">Klik disini</a>""", unsafe_allow_html=True)
        
        st.subheader("""Tipe Data""")
        st.write("""
        Tipe data yang di gunakan pada dataset ini adalah bertipe NUMERICAL.
        """)

    elif selected == "Dataset":
        st.subheader("""Dataset Saham """)
        df = pd.read_csv('https://raw.githubusercontent.com/Ihsan210702/uaspsd/main/RALS.JK.csv')
        st.dataframe(df, width=600)

    elif selected == "Prepocessing":
        st.subheader("""Preprocessing""")
        df = pd.read_csv('Data4Fitur_UASPSD.csv')
        df = df.iloc[:, 1:7]
        st.dataframe(df)
        st.subheader("""Normalisasi Data""")
        st.write("""Rumus Normalisasi Data :""")
        st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
        # df = pd.read_csv('https://raw.githubusercontent.com/HanifSantoso05/dataset_matkul/main/anemia.csv')
        st.markdown("""
        Dimana :
        - X = data yang akan dinormalisasi atau data asli
        - min = nilai minimum semua data asli
        - max = nilai maksimum semua data asli
        """)

        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaledX = scaler.fit_transform(df)
        features_namesX = df.columns.copy()
        #features_names.remove('label')
        scaled_featuresX = pd.DataFrame(scaledX, columns=features_namesX)

        st.subheader('Hasil Normalisasi Data')
        st.dataframe(scaled_featuresX.iloc[:,0:6], width=600)

    elif selected == "Modeling":

        df = pd.read_csv('Data4Fitur_UASPSD.csv')
        df = df.iloc[:, 1:7]

        scaler = MinMaxScaler()
        #scaler.fit(features)
        #scaler.transform(features)
        scaledX = scaler.fit_transform(df)
        features_namesX = df.columns.copy()
        #features_names.remove('label')
        scaled_featuresX = pd.DataFrame(scaledX, columns=features_namesX)

        #Split Data 
        training, test = train_test_split(scaled_featuresX.iloc[:,0:5],test_size=0.3, random_state=0,shuffle=False)#Nilai X training dan Nilai X testing
        training_label, test_label = train_test_split(scaled_featuresX.iloc[:,-1], test_size=0.3, random_state=0,shuffle=False)#Nilai Y training dan Nilai Y testing


        st.write("#### Percobaan Model Terbaik")
        st.markdown("""
        Dimana :
        - Jumlah Fitur Transform = [1,2,3,4,5,6,7] 
        - Test Size = [0.2,0.3,0.4]
        - K = [2,3,5,7,9]
        """)
        df_percobaan = pd.read_csv('Data4Fitur_UASPSD.csv')
        st.write('##### Hasil :')
        data = pd.DataFrame(df_percobaan.iloc[:,1:6])
        st.write(data)
        st.write('##### Grafik Pencarian Nilai Error Terkecil :')
        st.line_chart(data=data[['MSE_coba','MAPE_coba']], width=0, height=0, use_container_width=True)
        st.write('##### Model Terbaik Berdasarkan :')
        st.info("Jumlah Fitur = 5, K = 2, Test_Size = 0.3, Nilai Erorr MSE= 0.00718, Nilai Error MAPE = 0,061")
        st.write('##### Model KNN :')

        # load saved model
        with open('model_knnterbaik_pkl' , 'rb') as f:
            model = pickle.load(f)
        regresor = model.fit(training, training_label)
        st.info(regresor)

            

    elif selected == "Implementation":
        with st.form("Implementation"):
            df = pd.read_csv('Data_5_Fitur_Transform.csv')
            df = df.iloc[:, 1:7]
            scaler = MinMaxScaler()
            #scaler.fit(features)
            #scaler.transform(features)
            scaledX = scaler.fit_transform(df)
            features_namesX = df.columns.copy()
            #features_names.remove('label')
            scaled_featuresX = pd.DataFrame(scaledX, columns=features_namesX)

            #Split Data 
            training, test = train_test_split(scaled_featuresX.iloc[:,0:5],test_size=0.3, random_state=0,shuffle=False)#Nilai X training dan Nilai X testing
            training_label, test_label = train_test_split(scaled_featuresX.iloc[:,-1], test_size=0.3, random_state=0,shuffle=False)#Nilai Y training dan Nilai Y testing

            #Modeling
            # load saved model
            with open('model_knnterbaik_pkl' , 'rb') as f:
                model = pickle.load(f)
            regresor = model.fit(training, training_label)
            pred_test = regresor.predict(test)
            
            #denomalize data test dan predict
            hasil_denormalized_test = []
            for i in range(len(test)):
                df_min = df.iloc[:,0:5].min()
                df_max = df.iloc[:,0:5].max()
                denormalized_data_test_list = (test.iloc[i]*(df_max - df_min) + df_min).map('{:.1f}'.format)[0]
                hasil_denormalized_test.append(denormalized_data_test_list)

            hasil_denormalized_predict = []
            for y in range(len(pred_test)):
                df_min = df.iloc[:,0:5].min()
                df_max = df.iloc[:,0:5].max()
                denormalized_data_predict_list = (pred_test[y]*(df_max - df_min) + df_min).map('{:.1f}'.format)[0]
                hasil_denormalized_predict.append(denormalized_data_predict_list)

            denormalized_data_test = pd.DataFrame(hasil_denormalized_test,columns=["Testing Data"])
            denormalized_data_preds = pd.DataFrame(hasil_denormalized_predict,columns=["Predict Data"])

            #Perhitungan nilai error
            MSE = mean_squared_error(test_label,pred_test)
            MAPE = mean_absolute_percentage_error(denormalized_data_test,denormalized_data_preds)

            # st.subheader("Implementasi Prediksi ")
            v1 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 5 bulan sebelum periode yang akan di prediksi')
            v2 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 4 bulan sebelum periode yang akan di prediksi')
            v3 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 3 bulan sebelum periode yang akan di prediksi')
            v4 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 2 bulan sebelum periode yang akan di prediksi')
            v5 = st.number_input('Masukkan Jumlah barrel yang telah di import pada 1 bulan sebelum periode yang akan di prediksi')

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    v1,
                    v2,
                    v3,
                    v4,
                    v5,
                ])
                
                df_min = df.iloc[:,0:5].min()
                df_max = df.iloc[:,0:5].max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                st.write("#### Normalisasi data Input")
                st.write(input_norm)

                input_pred = regresor.predict(input_norm)

                st.write('#### Hasil Prediksi')
                st.info((input_pred*(df_max - df_min) + df_min).map('{:.1f}'.format)[0])
                st.write('#### Nilai Error')
                st.write("###### MSE :",MSE)
                st.write("###### MAPE :",MAPE)