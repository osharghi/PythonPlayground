
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydotplus


def read():

    df_x = pd.DataFrame({"X1": [0, 0, 1, 1, 1, 1, 0, 0], "X2": [0, 1, 0, 1, 0, 1, 1, 0], "X3": [0, 0, 1, 0, 0, 1, 1, 1]})
    df_y = pd.DataFrame({"Y": [1, 1, 0, 1, 1, 0, 0, 0]})

    #3 Display DF X and Y
    print('DATA FRAMES')
    print('\nDF X:')
    print(df_x.to_string())
    print('\nDF Y:')
    print(df_y.to_string())

    #4 Display DF column names
    x_columns = list(df_x.columns.values)
    y_columns = list(df_y.columns.values)

    print('\nCOLUMN NAMES')
    print('X: ' + ' '.join(x_columns))
    print('Y: ' + ' '.join(y_columns))

    #5 Split training and test data
    X_train, X_test, Y_train, Y_test = train_test_split(df_x, df_y, test_size=0.30)


    #6 Display Train and Test
    print("\n\nTRAINING SET")
    print('X TRAIN:')
    print(X_train)
    print('Y TRAIN:')
    print(Y_train)
    print("\n\nTEST SET")
    print('X TEST:')
    print(X_test)
    print('Y TEST:')
    print(Y_test)

    #7 Create Decision Tree Classifier
    dtree = DecisionTreeClassifier()

    #8 Fit DT
    dtree.fit(X_train, Y_train)

    #9 Predict DT
    Y_predict = dtree.predict(X_test)

    #10 Results
    print('\n\nRESULTS: ')
    print('Y_test:')
    print(Y_test)
    print('Y_predict:')
    print(Y_predict)

    dot_data = StringIO()
    export_graphviz(dtree,
                    out_file=dot_data,
                    feature_names=x_columns,
                    class_names=['0', '1'],
                    filled=True, rounded=True,
                    impurity=False)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("tree.png")


def main():
    read()


if __name__ == "__main__":
    main()
