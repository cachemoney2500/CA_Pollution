import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from geopy.distance import geodesic
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


#return a set of all measurement site numbers in a county
def extract_sites(county_data):
    return set(county_data["Site Num"].tolist())

def to_date(d):
    return dt.datetime.strptime(d,'%Y-%m-%d')

def add_points(start,end,fire_lat,fire_long):
    #only include dates from when fire was burning
    d_start = to_date(start)
    d_end = to_date(end)
    CO_fire = CO[CO["Date Local"]>=d_start]
    CO_fire = CO_fire[CO_fire["Date Local"]<=d_end]
    #only look at sites within +- 2 degrees of the lat/long
    CO_fire = CO_fire[CO_fire["Latitude"].between(fire_lat-2,fire_lat+2)]
    CO_fire = CO_fire[CO_fire["Longitude"].between(fire_long-2,fire_long+2)]

    #store latitude & longitude of each site & the peak CO reading from the duration of the fire
    distances = []
    peaks = []
    sites = extract_sites(CO_fire)
    for site in sites:
        site_data = CO_fire[CO_fire["Site Num"]==site]
        lat = site_data["Latitude"].tolist()[0]
        long = site_data["Longitude"].tolist()[0]
        peaks.append(np.nanmax(site_data["AQI"]))
        distances.append(geodesic((fire_lat,fire_long),(lat,long)).miles)

    return distances,peaks

def extract_data(fires):
    distances = []
    fire_sizes = []
    peaks = []
    #add data points for each fire
    for index, row in fires.iterrows():
        start = row["Start Date"]
        end = row["End Date"]
        fire_lat = row["Lattitude"]
        fire_long = row["Longitude"]
        fire_size = row["Acres Burned"]
        d,p = add_points(start,end,fire_lat,fire_long)
        distances += d
        peaks += p
        fire_sizes += [fire_size]*len(d)

    #form X and y from data
    #features in X are distance from fire, and size of fire
    #labels are the peak CO reading from the time frame of the fire
    X = np.zeros((len(distances),2))
    X[:,0] = np.array(distances)
    X[:,1] = np.array(fire_sizes)
    y = np.array(peaks)

    return X,y



if __name__ == "__main__":
    #read in dataset
    CO = pd.read_csv("CA_COdata_2009to2018.csv")
    fires = pd.read_csv("fires.csv")

    #convert date column to dt objects
    CO["Date Local"] =  pd.to_datetime(CO["Date Local"],format='%Y-%m-%d')


    #extract inputs & labels from the data set
    X_train,y_train = extract_data(fires[fires["Fire Name"]!="Station"])
    X_test,y_test = extract_data(fires[fires["Fire Name"]=="Station"])

    #fit a linear regression model to the data and predict for a single fire
    linReg = linear_model.LinearRegression()
    linReg.fit(X_train,y_train)
    y_pred = linReg.predict(X_test)
    print("Linear Regression mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
    print('Linear Regression variance score: %.2f' % r2_score(y_test, y_pred))


    #fit a RANSAC regression model to the data and predict for a single fire
    ransac = linear_model.RANSACRegressor(random_state=20)
    ransac.fit(X_train,y_train)
    y_pred_r = ransac.predict(X_test)
    print("RANSAC Regression mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred_r))
    print('RANSAC Regression variance score: %f' % r2_score(y_test, y_pred_r))

    plt.plot(X_test[:,0],y_pred,'r')
    plt.scatter(X_test[:,0],y_test)
    plt.xlabel("Distance from Fire")
    plt.ylabel("Peak CO Readings")
    plt.title("Station Fire Linear Model")

    plt.figure(2)
    plt.plot(X_test[:,0],y_pred_r,'r')
    plt.scatter(X_test[:,0],y_test)
    plt.xlabel("Distance from Fire")
    plt.ylabel("Peak CO Readings")
    plt.title("Station Fire RANSAC Model")
    plt.show()
