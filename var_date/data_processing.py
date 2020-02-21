import pandas as pd


def load_data():
    # use pandas to view the data structure
    col_names=['buying','maint','doors','persons','lug_boot','safety','class']
    data=pd.read_csv("./car.csv",names=col_names)
    return data

def convert2onehot(data):
    # covert data to onehot representation
    return pd.get_dummies(data,prefix=data.columns)

if __name__=="__main__":
    data=load_data()
    new_data=convert2onehot(data)
    print(data.head())
    print("\nNum of data:",len(data),'\n')
    for name in data.keys():
        print(name,pd.unique(data[name]))
    print("\n",new_data.head(2))
    new_data.to_csv('car_onehot.csv',index=False)