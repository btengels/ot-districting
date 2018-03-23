# To run this

  - `conda create -n districting python=3.6`  This creates a new virtual environment called "districting" but you can call it what you like
  - `source activate districting` Any python commands will look to this python build first
  - `conda install -c conda-forge cartopy`  This library is a pain and we have to install it this way
  - `pip install -r requirements.lock`  Get all the other libraries

  Now we can run the code!

  - `python main.py` or pip install jupyter and `run main.py` within an ipython session (this is what I usually do)
