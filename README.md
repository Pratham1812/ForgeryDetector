# To setup the project
 
Clone the repo and open the terminal in the desired directory.
Then run


```
python -m venv myvenv
```

```
pip install -r requirements.txt
```

```
python app.py 
```

Launch http://localhost:5000

Select original and upconverted videos to be tested and enter threshold value for edge detection.

# To run directly on Jupyter notebook

1. Open the notebook  - Final_Script_binary_classification.ipynb
2. Run all cells upto the Test section
3. In the test section, enter paths for original video and upconverted video
4. In the next cell, set edge_threshold value. For wavelet based detection the threshold is between 0 and 1. For other schemes the threshold is in order of hundreds.
5. Typical values of threshold for each of the operators are :
   Prewitt : 100
   Sobel : 100
   Kirsch : 300
   Wavelet : 0.05
7. Set n_frames to number of frames required to be plotted as interpolated/original in the binary classification plot.
