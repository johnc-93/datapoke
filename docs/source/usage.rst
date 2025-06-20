Usage
=====

Installation
------------

Install using pip:

.. code-block:: console

    (.venv) $ pip install datapoke


Loading data
------------

Datapoke uses the PokeFrame class as a wrapper for pandas dataframes.

Data can either be loaded by either calling PokeFrame on a pandas dataframe:

.. code-block:: python

    pf = PokeFrame(df)

Or from a csv by using PokeFrame.load_csv()

.. code-block:: python

    pf = PokeFrame.load_csv("some_csv_path.csv")