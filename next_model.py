import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from geopy.distance import geodesic
from geographiclib.geodesic import Geodesic as geodesic2
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

import pdb


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
    distances, bearings, peaks = [], [], []
    sites = extract_sites(CO_fire)
    for i, site in enumerate(sites):
        site_data = CO_fire[CO_fire["Site Num"]==site]
        lat = site_data["Latitude"].tolist()[0]
        long = site_data["Longitude"].tolist()[0]
        peaks.append(np.nanmax(site_data["AQI"]))
        distances.append(geodesic((fire_lat,fire_long),(lat,long)).miles)
        bearings.append(geodesic2.WGS84.Inverse(fire_lat,fire_long,lat,long)['azi1'])

    return distances,bearings,peaks

def extract_data(fires, with_bearings=True):
    distances, bearings,fire_sizes, peaks = [], [], [], []
    #add data points for each fire
    for index, row in fires.iterrows():
        start = row["Start Date"]
        end = row["End Date"]
        fire_lat = row["Lattitude"]
        fire_long = row["Longitude"]
        fire_size = row["Acres Burned"]
        d,b,p = add_points(start,end,fire_lat,fire_long)
        distances += d
        bearings += b
        peaks += p
        fire_sizes += [fire_size]*len(d)

    #form X and y from data
    #features in X are distance from fire, and size of fire
    #labels are the peak CO reading from the time frame of the fire
    if with_bearings: X = np.zeros((len(distances),3))
    else: X = np.zeros((len(distances),2))
    
    X[:,0] = np.array(distances)
    X[:,1] = np.array(fire_sizes)
    if with_bearings: X[:,2] = np.array(bearings)
    y = np.array(peaks)

    return X,y



if __name__ == "__main__":
    #read in dataset
    CO = pd.read_csv("CA_COdata_2009to2018.csv")
    CO["Date Local"] =  pd.to_datetime(CO["Date Local"],format='%Y-%m-%d')

    fires = pd.read_csv("fires_summer.csv")
    

    
#        extract inputs & labels from the data set
 #for comparing model performance with and without bearings for all fires
    for test_fire in fires['Fire Name']:
        for bearings in [True]:
            print('\n%s Fire, bearings = %i' %(test_fire, bearings))
            X_train,y_train = extract_data(fires[fires["Fire Name"]!=test_fire], with_bearings=bearings)
            X_test,y_test = extract_data(fires[fires["Fire Name"]==test_fire], with_bearings=bearings)
        
            # map polynomial features
            feat_map = PolynomialFeatures(degree=2)
            X_train_ = feat_map.fit_transform(X_train)
            X_test_ = feat_map.fit_transform(X_test)
            
            
            #fit a linear regression model to the data and predict for a single fire
            linReg = linear_model.LinearRegression()
            linReg.fit(X_train_,y_train)
            y_pred = linReg.predict(X_test_)
            print("Linear Regression MSE: %.2f"
              % mean_squared_error(y_test, y_pred))
        #            print('Linear Regression variance score: %.2f' % r2_score(y_test, y_pred))
            
                   
            plt.scatter(X_test[:,2],y_pred,c='r')
            if bearings:
                plt.scatter(X_test[:,2],y_test, c=X_test[:,0], cmap='viridis')
                plt.colorbar()
            else:
                plt.scatter(X_test[:,0],y_test)

            plt.xlabel("Distance from Fire")
            plt.ylabel("Peak CO Readings")
            plt.title("%s Fire Linear Model" %test_fire)
            
            #plt.show()
            
            
            #fit a RANSAC regression model to the data and predict for a single fire
#            ransac = linear_model.RANSACRegressor(random_state=20)
#            ransac.fit(X_train,y_train)
#            y_pred_r = ransac.predict(X_test)
#            print("RANSAC Regression MSE: %.2f"
#              % mean_squared_error(y_test, y_pred_r))
#                    print('RANSAC Regression variance score: %f' % r2_score(y_test, y_pred_r))
                        
#            plt.figure(2)
#            plt.plot(X_test[:,0],y_pred_r,'r')
#            if bearings:
#                plt.scatter(X_test[:,0],y_test, c=X_test[:,2], cmap='viridis')
#                plt.colorbar()
#            else:
#                plt.scatter(X_test[:,0],y_test)
#            plt.xlabel("Distance from Fire")
#            plt.ylabel("Peak CO Readings")
#            plt.title("%s Fire RANSAC Model" %test_fire)
            

