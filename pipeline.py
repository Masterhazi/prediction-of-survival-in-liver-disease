import pandas as pd
import numpy as np

class LiverSurvivalPipeline2:

    def __init__(self, model, feature_columns, medians):
        self.model = model
        self.feature_columns = feature_columns
        self.medians = medians

    def bilirubin_score(self, bilirubin):
        if bilirubin < 2:
            return 1
        elif 2 <= bilirubin <= 3:
            return 2
        else:
            return 3

    def albumin_score(self, albumin):
        if albumin > 3.5:
            return 1
        elif 2.8 <= albumin <= 3.5:
            return 2
        else:
            return 3

    def ascites_score(self, ascites):
        if ascites == 'N':
            return 1
        else:
            return 3

    def prothrombin_score(self, prothrombin):
        if prothrombin < 4:
            return 1
        elif 4 <= prothrombin <= 6:
            return 2
        else:
            return 3
    
    def preprocess(self,df):
        df = df.copy()

        df['age_in_years'] = df['Age']/365.25
        df.drop('Age', axis = 1, inplace = True)

        cat_cols = ['Drug','Sex','Ascites','Hepatomegaly','Spiders','Edema']
        for col in cat_cols:
            df[col] = df[col].fillna('unknown')

        numeric_cols = [
            'Bilirubin','Cholesterol','Albumin','Copper',
            'Alk_Phos','SGOT','Tryglicerides',
            'Platelets','Prothrombin','Stage'
        ]

        for col in numeric_cols:
            df[f'{col}_missing'] = df[col].isna().astype(int)
            df[col] = df[col].fillna(self.medians[col])

        df['Thrombocytopenia'] = np.where(df['Platelets'] < 150, 1, 0)

        df['el_bil'] = np.where(
            (df['Bilirubin'] > 0.2) & (df['Bilirubin'] < 1.3), 0, 1
        )

        df['lo_alb'] = np.where(df['Albumin'] < 3.5, 1, 0)
        df['el_co'] = np.where(df['Copper'] > 140, 1, 0)
        df['el_phos'] = np.where(df['Alk_Phos'] > 147, 1, 0)
        df['el_sgot'] = np.where(df['SGOT'] > 45, 1, 0)
        df['el_clot'] = np.where(df['Prothrombin'] > 13.5, 1, 0)

        df['Bilirubin_Score'] = df['Bilirubin'].apply(self.bilirubin_score)
        df['Albumin_Score'] = df['Albumin'].apply(self.albumin_score)
        df['Ascites_Score'] = df['Ascites'].apply(self.ascites_score)
        df['Prothrombin_Score'] = df['Prothrombin'].apply(self.prothrombin_score)

        df['Child_Pugh_Score'] = df[
            ['Bilirubin_Score','Albumin_Score',
             'Ascites_Score','Prothrombin_Score']
        ].sum(axis=1)

        drop_cols = ['id', 'id_missing', 'N_Days', 'N_Days_missing', 'Status','Bilirubin_Score', 'Albumin_Score', 'Ascites_Score',
       'Prothrombin_Score', 'Age', 'target_missing', 'Stage_missing']

        df.drop(drop_cols, axis=1, inplace=True, errors='ignore')

        df = pd.get_dummies(df, drop_first= True)

        df = df.reindex(columns=  self.feature_columns, fill_value = 0)

        return df

    def predict_proba(self,df):
        df_processed = self.preprocess(df)
        return self.model.predict_proba(df_processed)[:,1]
    
    def predict(self, df, threshold = 0.5):
        probs = self.predict_proba(df)
        return (probs>= threshold).astype(int)
