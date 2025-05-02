import pandas as pd
import logging

logging.basicConfig(level = logging.INFO)

csv_path = r"C:\Users\Olive\Downloads\Pizza+Place+Sales\pizza_sales\order_details.csv"

try:
    df = pd.read_csv(csv_path, encoding = "utf-8")
    logging.info("CSV loaded as utf-8")
except UnicodeDecodeError:
    logging.warning("CSV load encoding error - falling back to latin1")
    #some nicer encoding handling here
    df = pd.read_csv(csv_path, enconding = "latin1")
    logging.info("CSV loaded as latin1")

headers = df.columns.tolist()

inferred_types = df.dtypes.to_dict()

logging.info(inferred_types)


