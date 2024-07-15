import dill 
import joblib 
import pandas as pd 
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from category_encoders import HashingEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

#Scoring metric
scoring_metric = 'roc_auc'

#Targeted actions
target_actions = ['sub_car_claim_click', 'sub_car_claim_submit_click', 'sub_open_dialog_click', 'sub_custom_question_submit_click', 'sub_call_number_click', 'sub_callback_submit_click', 'sub_submit_success', 'sub_car_request_submit_click']
def target_func(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['target_action'] = 0
    df.loc[df['event_action'].isin(target_actions), 'target_action'] = 1
    df['event_action'] = df['target_action']
    return df.drop(columns = 'target_action')

#Filtering data
def filter_data(df : pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = ['device_model', 'event_value', 'visit_date', 'session_id', 'event_category', 'hit_type', 'hit_time', 'utm_keyword', 'client_id', 'device_brand']
    return df.drop(columns = cols_to_drop)

#Adding new features or replacing the existing ones with more advantageous ones
def add_features(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hit_date"] = df["hit_date"].astype("datetime64[ns]").dt.month
    return df

# Dropping rows with NaNs in specific columns
def dropna_specific(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=['device_os', 'hit_referer', 'event_label'])

#Filling up missing values utm_source, utm_campaign and utm_adcontent
def corr_filling_utms(df : pd.DataFrame) -> pd.DataFrame:
    columns_to_change = ['utm_source', 'utm_adcontent', 'utm_campaign']
    df = df.copy()
    def filler_function(target_column):
        filler_col = df.groupby(['utm_medium'])[target_column].value_counts().groupby(level = 0).nlargest(1).index

        medium_col = []
        target_col = []

        for i in filler_col:
            medium_col.append(i[1])
            target_col.append(i[2])

        medium_to_target_map = dict(zip(medium_col, target_col))
        df[target_column].fillna(df['utm_medium'].map(medium_to_target_map), inplace = True)
    for col in columns_to_change:
        filler_function(col)
    return df

#reducing the number of unique value in visit_time column
def categorize_visit_time(df : pd.DataFrame) -> pd.DataFrame : 
    import pandas as pd
    def categorize_time(hour):
        if 0 <= hour < 12:
            return "Morning"
        elif 12 <= hour < 18:
            return "Afternoon"
        else:
            return "Evening"
    df['visit_time'] = pd.to_datetime(df['visit_time'], format='%H:%M:%S')    
    df['visit_time'] = df['visit_time'].dt.hour.apply(categorize_time)

    return df


#Dropping remnant NAN values in the dataset
def dropna_final(df : pd.DataFrame) -> pd.DataFrame:  
    return df.dropna()

def pipeline() -> None :
    print('User action predictor model pipeline')

    #Data loading
    df_hits = pd.read_csv(filepath_or_buffer = "./data/ga_hits.csv", low_memory = False)
    df_session = pd.read_csv(filepath_or_buffer = './data/ga_sessions.csv', low_memory = False)
    df = pd.merge(left = df_hits, right = df_session, on = "session_id")

    # Preprocess the entire DataFrame first
    df = target_func(df)
    df = filter_data(df)
    df = dropna_specific(df)
    df = corr_filling_utms(df)
    df = dropna_final(df)
    print(df.shape)

    X = df.drop(columns = 'event_action')
    y = df['event_action']

    numerical_features = ['hit_date', 'hit_number', 'visit_number']
    ohe_cols = ['visit_time', 'device_category', 'device_os', 'device_browser']
    hash_cat = ['hit_referer', 'event_label', 'hit_page_path', 'utm_source', 'utm_campaign', 'utm_adcontent', 'device_screen_resolution', 'geo_country', 'geo_city', 'utm_medium']

    #Feature engineering
    numerical_transformer = Pipeline(steps = [
        ('scaler', StandardScaler())
    ])
    
    ohe_transformation = Pipeline(steps = [
        ('ohe', OneHotEncoder(handle_unknown = 'ignore'))
    ])
    hasher = Pipeline(steps = [
        ('hasher', HashingEncoder())
    ])

    column_transformer = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('ohe_transformation', ohe_transformation, ohe_cols),
        ('hashing', hasher, hash_cat)
    ])

    #preprecessor for pipelines
    preprocessor = Pipeline(steps = [
        ('utm_filler', FunctionTransformer(corr_filling_utms)),
        ('feature_add', FunctionTransformer(add_features)),
        ('visit_time_categorizer', FunctionTransformer(categorize_visit_time)),
        ('column_transformer', column_transformer)
    ])

    models = [
        MLPClassifier(),
        RandomForestClassifier(),
    ]

    best_score = .0
    best_pipe = None 
    for model in models:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        score = cross_val_score(pipe, X, y, cv = 4, scoring = scoring_metric, error_score = 'raise')
        print(f'model : {type(model).__name__}, {scoring_metric}_mean:{score.mean():.4f}, {scoring_metric}_std : {score.std():.4f}')
        if score.mean() > best_score : 
            best_score = score.mean()
            best_pipe = pipe
    
    print(f'best model: {type(best_pipe.named_steps["classifier"]).__name__}, {scoring_metric}: {best_score:.4f}')

    #Fitting perfect pipeline for whole dataset
    best_pipe.fit(X = X, y = y)
    model_filename = f'./models/user_action.pkl'
    dill.dump({'model' : best_pipe,
        'metadata' :{
            'name' : 'User action predictor',
            'author' : 'Umidjon Sattorov',
            'version' : 1,
            'date' : datetime.now(),
            'type' : type(best_pipe.named_steps['classifier']).__name__,
            'accuracy' : best_score
        }
    }, open('./models/user_action_predictor_1.pkl', 'wb'))

    print(f'Model is saved as {model_filename} in models directory')


if __name__ == '__main__':
    pipeline()
