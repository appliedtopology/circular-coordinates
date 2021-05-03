# Circular-Coordinates
## _Circular coordinates from persistent cohomology_





Circular-Coordinates is an all in one class that facilitates creating and plotting circular coordinates from persistent cohomology functions hidden in your dataset.  

## Features

- takes input data in the form of numpy array or pandas dataframe
- utilizes the ripser library for fast persistent cohomology barcode calculation 
- Provides multiple ways of plotting and visualizing the output.

## Setup
You can install the library directly from github(will be published on Pypi soon). You will need PyQt5 before installing(only for visualizing plots in ide will be optional in future).

```
pip install PyQt5
pip install git+https://github.com/appliedtopology/circular-coordinates
```

## Example usage
Circular coordinates can be calculated and visualized with only a few lines of code. The circular coordinates are outputed mapped between [0,1]. When visualising the coordinates are denoted by colors on the color wheel (with its values mapped to [0,1]). Thus forming a loop stretching from 0 to 1.
```
import pandas as pd
import circularcoordinates

df = pd.read_csv('malaria.csv')
features = df[['Weight', 'Glucose', 'Accuri', 'RBC']]

prime=11
circ=circularcoordinates.circular_coordinate(prime)
vertex_values=circ.fit_transform(features)
circ.plot_pca(features,vertex_values)
```
![PCA PLOT](https://drive.google.com/uc?export=download&id=16BpwdQOkTnehwbRLc1SYpDiJ2Mmm_PaO)

If we already have the ripser output dictionary we can directly compute the circular coordinates without recomputing the dictionary.
```
ripser_output=circ.rips
vertex_values=circ.circular_coordinate(ripser_output)
```
We can also plot the persistant homology barcodes
```
circ.plot_barcode(circ.rips['dgms'][1])
```
![Barcode PLOT](https://drive.google.com/uc?export=download&id=1ARj-ta2Zk-pVN62l6_OjfzyMUouwTkDA)

Circular coordinates can also be plottted against externa data to see what patterns emerge.
```
circ.plot_eps(df['Day Post Infection'],vertex_values)
```
![Barcode PLOT](https://drive.google.com/uc?export=download&id=1bzL3k6QmCYeSKyNetpHewsT8zbAqURpv)

## License

MIT
