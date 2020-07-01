# DCF-PM2.5
Dynamic calibration framework for low-cost PM2.5 sensors<br>
<span style="color:#007bff">```DCF```</span>
<span style="color:#17a2b8">```PM25```</span>
<span style="color:#6c757d">```Calibration```</span>
<span style="color:#28a745">```Open```</span>

**Dynamic Calibration Model Status Report**: <https://pm25.lass-net.org/DCF/><br>
**AirBox Status Report**: <https://pm25.lass-net.org/AirBox/><br>
**PM2.5 Open Data Portal**: <https://pm25.lass-net.org/>

[Model Info](#info) | [Requires](#require) | [Usage](#usage) | [Resource](#resource) | [Troubleshooting](#trouble)


<a name="info"></a>
## Model Info

- **Data Length**:

   7, 14, 21, 31 days

- **Feature**: 

   PHTR, PHT, PHR, PTR, PH, PT, PR, P

   |   | PM2.5 | Hour at which data is sensed | Temperature | Relative humidity |
   |:--:|:--:|:--:|:--:|:--:|
   |**abbreviation**| P | H | T | R |
   |**type**| float | int | float | float |
   |**example**|11.3|1|32.5|72.5|

- **Method**:

   RR, LassoR, LinearR, BR, RFR, SVR, GAM
   
   |abbreviation|full name|package|
   |:-:|:--|:--|
   |RR|Ridge Regression|```sklearn.linear_model.Ridge```|
   |LassoR|Lasso Regression|```sklearn.linear_model.Lasso```|
   |LinearR|LinearRegression|```sklearn.linear_model.LinearRegression```|
   |BR|BayesianRidge|```sklearn.linear_model.BayesianRidge```|
   |RFR|Random Forest Regression|```sklearn.ensemble.RandomForestRegressor```|
   |SVR|Support Vector Regression|```sklearn.svm.SVR```|
   |GAM|Generalized Additive Model|```pygam```|

<a name="require"></a>
## Requires

**DCF training dependencies**

```Python3.6```

| Package | Version | Link |
|:------- |:-------:|:---- |
| joblib  | 0.15.1  | <https://joblib.readthedocs.io> |
| pygam   | 0.8.0   | <https://pygam.readthedocs.io/en/latest/> |
| scikit-learn | 0.23.1 | <https://scikit-learn.org/stable/> |

<a name="usage"></a>
## Usage

`Python3` `joblib` `pygam` `scikit-learn` `numpy` `pandas`

### find the nearest model
`math` `pandas`

```Python
import pandas as pd
import math

DegreeToRadians = lambda degree: degree * math.pi / 180
def distanceWithCoordinates(lat1, lon1, lat2, lon2):
    RADIUS = 6371
    dlat, dlon = DegreeToRadians(lat2-lat1), DegreeToRadians(lon2-lon1)
    lat1, lat2 = DegreeToRadians(lat1), DegreeToRadians(lat2)
    _ = (math.sin(dlat/2) ** 2) + (math.sin(dlon/2) ** 2) * math.cos(lat1) * math.cos(lat2)
    return RADIUS * 2 * math.atan2(math.sqrt(_), math.sqrt(1-_))
        
def find_site( device_lon, device_lat ):
    daily_status_url = "https://raw.githubusercontent.com/IISNRL/DCF-PM2.5/master/2020/20200620/20200620-PMS5003.json"
    models_info = pd.read_json(daily_status_url)

    CountingDistance = lambda row: distanceWithCoordinates( row['Latitude'], row['Longitude'], device_lat, device_lon )
    models_info[ 'distance' ] = models_info.apply( CountingDistance, axis=1 )
    models_sort = models_info.dropna( subset=['distance'] ).sort_values( by="distance" )
    
    return models_sort['site'][0], models_sort['distance'][0]
    
site, distance = find_site( 120.69, 23.99 )
```

### download model/config
`json` `requests` `urllib`

```Python
import json
import requests
import urllib

## config 
config_url = "https://raw.githubusercontent.com/IISNRL/DCF-PM2.5/master/2020/20200620/20200620-PMS5003-nantou.config"
r = requests.get(config_url)
content = r.content
config_dict = json.loads(content)

## joblib
model_url = "https://github.com/IISNRL/DCF-PM2.5/raw/master/2020/20200620/20200620-PMS5003-nantou.joblib"
urllib.request.urlretrieve(model_url, '20200620-PMS5003-nantou.joblib')
```

### load and predict
Import requirement packages

```Python
import pandas as pd
import numpy as np

import sklearn
import pygam
import joblib
```

Data preprocessing

```bash
# raw_value (Dict)
{'P': 11, 'H': 1, 'T': 32.5, 'R': 72.5}
# raw_DF (DataFrame)
raw_DF = pd.DataFrame([{'P': 11, 'H': 1, 'T': 32.5, 'R': 72.5},{'P': 12, 'H': 2, 'T': 33.1, 'R': 75},{'P': 15.6, 'H': 3, 'T': 32.1, 'R': 70.3}])
#       P    H     T     R
# 0  11.0  1.0  32.5  72.5
# 1  12.4  2.0  33.1  75.0
# 2  15.6  3.0  32.1  70.3
```

```Python
## single datapoint
# select fields
X_test = [raw_value[field] for field in config_dict["Feature"]]
# reshape
X_test = np.array(X_test).reshape(1, -1)

## series datapoints
# select fields
X_test = raw_DF[ list(config_dict["Feature"]) ]
```

Load calibration model and predict value

```Python
## load
lm = joblib.load( "20200620-PMS5003-nantou.joblib" )
## calibration
# "X_test" columns order should be the same as "Feature" in config
Y_pred = lm.predict( X_test )

## result
# array([8.77330524])
# array([ 8.77330524,  8.9214365 , 10.91355547])
```



<a name="resource"></a>
## Resource

### Realtime
[Calibration Models Status Report](https://pm25.lass-net.org/DCF/)

- **status:**<br>
   <https://pm25.lass-net.org/DCF/latest.json>
- **model file:** <br>
   latest-PMS5003-\<sitename\>.joblib<br>
   <https://pm25.lass-net.org/DCF/model/latest-PMS5003-nantou.joblib>
- **model config:**<br>
   latest-PMS5003-\<sitename\>.config<br>
   <https://pm25.lass-net.org/DCF/model/latest-PMS5003-nantou.config>
   
### Opendata
[ISNRL/DCF-PM2.5](https://github.com/IISNRL/DCF-PM2.5/tree/master/)

- **daily forlder:**<br>
   \<YYYY>/\<YYYYMMDD><br>
   <https://github.com/IISNRL/DCF-PM2.5/tree/master/2020/20200620>
- **daily status:**<br>
   \<YYYYMMDD\>-PMS5003.json<br>
   <https://github.com/IISNRL/DCF-PM2.5/blob/master/2020/20200620/20200620-PMS5003.json>
- **model file:** <br>
   \<YYYYMMDD\>-PMS5003-\<sitename\>.joblib<br>
   <https://github.com/IISNRL/DCF-PM2.5/blob/master/2020/20200620/20200620-PMS5003-nantou.joblib>
- **model config:**<br>
   \<YYYYMMDD\>-PMS5003-\<sitename\>.config<br>
   <https://github.com/IISNRL/DCF-PM2.5/blob/master/2020/20200620/20200620-PMS5003-nantou.config>
   

<a name="trouble"></a>
## Troubleshooting
**1. SVC - AttributeError**
   
It can happen that joblib fails to predict when model is based on `SVR`, for instance:
   
```bash
  File "/home/ubuntu/.pyenv/versions/3.7.5/lib/python3.7/site-packages/sklearn/svm/_base.py", line 317, in predict
    return predict(X)
  File "/home/ubuntu/.pyenv/versions/3.7.5/lib/python3.7/site-packages/sklearn/svm/_base.py", line 335, in _dense_predict
    X, self.support_, self.support_vectors_, self._n_support,
AttributeError: 'SVR' object has no attribute '_n_support'
```
In this case it is beacause in this project model training with `scikit-learn 0.23.1`. While models saved using one version of scikit-learn and be loaded in other versions, this may not be supported. Please see [sk-learn pickle](https://scikit-learn.org/dev/modules/model_persistence.html?highlight=pickle#security-maintainability-limitations) for more information. 

Please update the versions of scikit-learn and its dependencies:

- check deatails of installed package

```bash
pip3 show scikit-learn
```
- upgrade python package

```bash
pip3 install -U scikit-learn
```

***2. Other Error Message?**

please let us know if you get any further question.
