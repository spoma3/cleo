import csv
from Dictionaries import Light_Intensity_Dict, Wavelength_Dict
import pandas as pd

for key, value in Light_Intensity_Dict.items():
    df = pd.DataFrame(list(value.items()), columns=['Parameter', 'Value'])
    df.to_csv(f'imaging/Light_Intensity_Params/{key}.csv', sep='\t', index=False)

for key, value in Wavelength_Dict.items():
    df = pd.DataFrame(list(value.items()), columns=['Parameter', 'Value'])
    df.to_csv(f'imaging/Wavelength_Dict/{key}.csv', sep='\t', index=False)