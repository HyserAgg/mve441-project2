import pandas as pd


def main():
    df = load_SEQ_data()
    print(df)


def load_SEQ_data() -> pd.DataFrame:
    if exists('data/data.h5'):
        feature_df = pd.read_hdf('data/data.h5')
        label_df = pd.read_csv('data/labels.csv')
        feature_df = feature_df.iloc[:,1:]
        return label_df["Class"], feature_df.iloc[:,1:]  
    else:
        feature_df = pd.read_csv("data/data.csv")
        label_df = pd.read_csv('data/labels.csv')
        feature_df.to_hdf('data/data.h5',key = 'df', mode='w') 
        return label_df["Class"], feature_df.iloc[:,1:]  

if __name__ == "__main__":
    main()