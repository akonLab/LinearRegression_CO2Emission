import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

fuel_dataset = pd.read_csv('CO2 Emissions_Canada.csv')


def linear_reg(feature_name1, feature_name2):
    visual_prediction_on_graph(feature_name1, feature_name2)
    plt.show()

    line = LinearRegression()
    x = pd.DataFrame(fuel_dataset[feature_name1])
    y = pd.DataFrame(fuel_dataset[feature_name2])
    line.fit(x, y)

    visual_prediction_on_graph(feature_name1, feature_name2)
    plt.plot(x, line.predict(x), color='red')
    plt.show()

    print("Данные между 2 свойствами (", feature_name1, ",", feature_name2, ")")
    moreData(line, x, y)


def visual_prediction_on_graph(feature_name1, feature_name2):
    plt.figure(figsize=(10, 6))
    plt.scatter(fuel_dataset[feature_name1], fuel_dataset[feature_name2], alpha=0.3)
    plt.xlabel(feature_name1)
    plt.ylabel(feature_name2)
    plt.xlim(0, fuel_dataset[feature_name1].max() + 5)
    plt.ylim(0, fuel_dataset[feature_name2].max() + 5)


def moreData(line, x, y):
    print("Точность результата (coefficient of determination): ", line.score(x, y))
    print("ваша модель предсказывает реакцию, когда x = 0 (intercept): ", line.intercept_)
    print("прогнозируемый ответ увеличивается, когда x увеличивается на единицу (scope): ", line.coef_)
    print("прогнозируемый ответ (prediction): ", line.predict(x))


def colorTable():
    sns.heatmap(fuel_dataset.corr(), cmap='coolwarm', annot=True)
    plt.show()


def liner_reg_to_all_feautures():
    # removing feautures with non numeric data
    fuel = fuel_dataset[
        ['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Comb (L/100 km)',
         'CO2 Emissions(g/km)']]
    fuel.head()

    y = fuel['CO2 Emissions(g/km)']
    x = fuel.drop('CO2 Emissions(g/km)', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    clf = LinearRegression()
    clf.fit(x_train, y_train)
    yhat_test = clf.predict(x_test)
    yhat_train = clf.predict(x_train)

    print("Test Accuracy : ", r2_score(yhat_test, y_test))
    print("Training Accuracy : ", r2_score(yhat_train, y_train))
    print("Test Coefficient determination:", clf.score(x_test, y_test))
    print("Training Coefficient determination:", clf.score(x_train, y_train))
    print("Intercept:", clf.intercept_)
    print("Slope:", clf.coef_)


if __name__ == '__main__':
    colorTable()

    # linear regression between 2 features
    linear_reg("CO2 Emissions(g/km)", "Fuel Consumption Comb (L/100 km)")

    # train 75%; test 25%
    print("\n Множественная линейная регрессия")
    liner_reg_to_all_feautures()
