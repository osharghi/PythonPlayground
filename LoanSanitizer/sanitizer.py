#
# author : Omid Sharghi (omid.sharghi@sjsu.edu)
#

from pandas import ExcelWriter
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

def sanitize_and_scale(file_name):

    if file_name.endswith('.csv'):
        df = pd.read_csv(file_name)
    elif file_name.endswith('.xlsx'):
        df = pd.read_excel(file_name)

    #start new Code
    df = df.drop(['Gender'], axis=1)
    df = df.dropna(subset=['ApplicantIncome', 'LoanAmount'])
    avg_approved = df[df.Loan_Status == 'Y'][['ApplicantIncome']].mean()


    for i in df.index:
        if pd.isnull(df.loc[i, 'Credit_History']) and (df.loc[i, 'Loan_Status'] == 'Y'):
            df.at[i, 'Credit_History'] = '1'

    # for i in df.index:
    #     if pd.isnull(df.loc[i, 'Married']) and (df.loc[i, 'CoapplicantIncome'] != 0):
    #         df.at[i, 'Married'] = 'Yes'

    # for i in df.index:
    #     if pd.isnull(df.loc[i, 'Credit_History']) and (df.loc[i, 'Loan_Status'] == 'N'):
    #         df.at[i, 'Credit_History'] = '0'

    # for i in df.index:
    #     if pd.isnull(df.loc[i, 'Dependents']) and (
    #             df.loc[i, 'Married'] == 'Yes' and df.loc[i, 'Property_Area'] == 'Rural'):
    #         df.at[i, 'Dependents'] = '2'

    df['CreditHistory_By_ApplicantIncome'] = df['Credit_History']/df['ApplicantIncome']
    df['CreditHistory_By_LoanAmount'] = df['Credit_History'] / df['LoanAmount']
    # df['Loan_To_Income_Ratio'] = df['LoanAmount'] / (df['ApplicantIncome'] + df['CoapplicantIncome'])

    #end new code


    df= df.dropna(subset=['Married', 'Dependents', 'Self_Employed', 'Credit_History', 'LoanAmount'])

    marriage_mapping = {'No': 0, 'Yes': 1}
    df['Married'] = df['Married'].map(marriage_mapping)

    loan_status_mapping = {'N': 0, 'Y': 1}
    df['Loan_Status'] = df['Loan_Status'].map(loan_status_mapping)

    education_mapping = {'Not Graduate': 0, 'Graduate': 1}
    df['Education'] = df['Education'].map(education_mapping)

    property_mapping = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}
    df['Property_Area'] = df['Property_Area'].map(property_mapping)

    self_employed_mapping = {'No': 0, 'Yes': 1}
    df['Self_Employed'] = df['Self_Employed'].map(self_employed_mapping)

    df['Dependents'] = df['Dependents'].replace('3+', 3)

    cols = ['Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

    df[cols] = df[cols].applymap(np.int64)

    df = df.dropna()
    df = df.drop(['Loan_ID'], axis=1)

    scaler = MinMaxScaler(feature_range=(0., 1.))
    X_scaled = scaler.fit_transform(df)
    col_names = df.columns
    df_scaled = pd.DataFrame(X_scaled, columns=col_names)

    output_sanitized_file(df_scaled)

    return df_scaled

def output_sanitized_file(df):
    writer = ExcelWriter('SanitizedData.xlsx')
    df.to_excel(writer, 'Sheet1')
    writer.save()
