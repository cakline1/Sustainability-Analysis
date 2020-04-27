def distance_start_finish(data_df, start_lat, start_lon, end_lat, end_lon):
    '''
    Return distances for data provided in data_df
    
    Arguments:
    data_df: a Pandas DataFrame that includes columns or starting and ending latitude and longitude.
                            
    start_lat: starting latitude
    
    start_lon: starting longitude
    
    end_lat: ending latitude
    
    end_lon: ending longitude
    
    '''
    import pandas as pd
    import numpy as np

    def haversine(row):
        from math import radians

        lon1 = float(row[start_lon])
        lat1 = float(row[start_lat])
        lon2 = float(row[end_lon])
        lat2 = float(row[end_lat])
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km

    return (data_df.apply(lambda row: haversine(row), axis=1) * 0.621371).tolist()