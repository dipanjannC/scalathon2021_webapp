#import streamlit
import streamlit as st
#st.beta_set_page_config(page_title='Scalathon2021')
st.set_page_config(layout='wide')
from streamlit import cli as stcli



# import other libraries
import numpy as np
import pandas as pd
import sys
from PIL import Image
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))




@st.cache(suppress_st_warning=True)
def get_dataset(dataset_name):

    if dataset_name == "Customer":
        st.spinner()
        with st.spinner(text='Loading Data ....'):
            data = pd.read_excel("/app/data/Customer_v2.xlsx",engine='openpyxl')
            st.success('Done')
        
    elif dataset_name == 'Reviews':
        st.spinner()
        with st.spinner(text='Loading Data ....'):
            data = pd.read_excel("/app/data/Reviews.xlsx",engine='openpyxl')
            st.success('Done')
        
    elif dataset_name == 'Sales Train':
        st.spinner()
        with st.spinner(text='Loading Data ....'):
            data = pd.read_excel('/app/data/Sales.xlsx', sheet_name='Train',engine='openpyxl')
            st.success('Done')
    elif dataset_name == 'Sales Test':
        st.spinner()
        with st.spinner(text='Loading Data ....'):
            data = pd.read_excel('/app/data/Sales.xlsx', sheet_name='Test',engine='openpyxl')
            st.success('Done')
        

    X = data.shape[0]
    y = data.shape[1]
    head = data.head(10)
    return X,y,head

def read_data(filename):
    df = pd.read_csv(filename)
    return df


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

# def get_classifier(clf_name, params):
#     clf = None
#     if clf_name == 'SVM':
#         clf = SVC(C=params['C'])
#     elif clf_name == 'KNN':
#         clf = KNeighborsClassifier(n_neighbors=params['K'])
#     else:
#         clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
#             max_depth=params['max_depth'], random_state=1234)
#     return clf



def main():
# Your streamlit code
    #header
    st.title('Advanced Scalathon 2021')

    # set homepage image
    st.text(' - submission by Annwesha and Dipanjan')
    st.write('\n')
    st.image('/app/data/img/homepage.jpg')
    st.subheader('\n  Problem Description') 
    st.markdown('XYZ is an American bookseller. It is a Fortune 1000 company and the bookseller with the largest number of retail outlets in the United States. It also sells books through various ecommerce channels. They sell around 3684 unique books in their different stores among which there are  4 top selling books on the basis of customer reviews. So XYZ has approached Bridgei2i Analytics Solutions to help them plan their growth strategy.')
    
    #st.beta_container()
    #sidebar
    st.sidebar.header('Home')
    dataset_name = st.sidebar.selectbox("Select Dataset",('Customer','Reviews','Sales Train','Sales Test'))
    #classifier_name = st.sidebar.selectbox("Select Classifier",('KNN','KMeans','Random Forest'))
    #params = add_parameter_ui(classifier_name)
    #clf = get_classifier(classifier_name, params)


    with st.beta_expander('Home'):
        st.write('A basic UI to support our evaluations')

    with st.beta_expander('Datasets'):
        #View Datasets
        X, y , head = get_dataset(dataset_name)
        st.write('Shape of dataset: ',X,y)
        st.write(" Data Sample :")
        st.table(head)
        if dataset_name == 'Reviews':
            st.write("WordCloud")
            st.image('/app/data/img/wordcloud.png')   

    with st.beta_expander('Data Profiling Report'):
        st.write("Data Profiling : ")
        st.write(read_data('/app/data/Profiling_Report_All_Columns.csv'))


    with st.beta_expander('Sentiment Analysis'):
        book_sentiments = read_data('/app/data/overall_book_sentiment.csv')
        book_sentiments = book_sentiments.rename(columns={'BookCode':'index'}).set_index('index')
        st.write('\n Overall Sentiment  Analysis of common 20 books ')
        st.image('/app/data/img/sentiment.png')
        st.write('\n Average Sentiment of common 20 books ')
        st.text('This would help us to find top 4 books')
        st.bar_chart(book_sentiments,use_container_width=True)
        

    with st.beta_expander('Topic Modelling'):
        st.markdown('Top 5 topics distribution across the 20 book codes \n')
        st.image('/app/data/img/topic_modelling.png')


    with st.beta_expander('Forecasting'):
        st.write('Juicy deets')
    
    with st.beta_expander('Question and Answers'):

        #reading files to be shown
        top4_books = read_data('/app/data/top4_books.csv')
        agegroups = read_data('/app/data/ageGroups.csv')
        sample_reviews = pd.read_csv('/app/data/sample_reviews.csv',skiprows=[1])

        #Question 1
        st.subheader('Question 1: ')
        st.markdown('XYZ wants to understand which books have most positive reviews among the whole set (They are expecting ranking for the top 4 books)')
        st.table(top4_books)
        st.write('\n\n')
        st.write(' Sample Reviews to back our results ')
        st.table(sample_reviews)

        #Question 2
        st.subheader('Question 2: ')
        st.markdown('XYZ also wants Bridgei2i to determine the profile of customers who have given the most positive reviews \n')
        st.write('\n\n')
        st.write('According to the Datasets Available we only can determine the Age Bracket of customers who are interested towards buying top 4 books')
        st.image('/app/data/img/cutomer_top4.png')
        st.table(agegroups)

    

    

    

if __name__ == '__main__':
    if st._is_running_with_streamlit:
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())


