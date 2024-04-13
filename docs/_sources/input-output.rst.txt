.. allinsql documentation master file, created by
   sphinx-quickstart on Sun Mar 17 22:07:11 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. _api.io:

============
Input/output
============



ClickHouse
~~~~~~~~~~~
.. autosummary::
   :toctree: dataframe/

   dataframe.readClickHouse

Spark
~~~~~~~~~
.. autosummary::
   :toctree: dataframe/

   dataframe.readSparkDf
   dataframe.DataFrame.toSparkDf



Csv
~~~~~~~~~
.. autosummary::
   :toctree: dataframe/

   dataframe.readCsv
   dataframe.DataFrame.toCsv


StarRocks
~~~~~~~~~
.. autosummary::
   :toctree: dataframe/

   dataframe.readStarRocks