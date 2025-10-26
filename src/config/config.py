import os
from flask.cli import load_dotenv

load_dotenv()

CSV_PATH = os.getenv("CSV_PATH")
DATE_COL = os.getenv("DATE_COL")
COMPANY_COL = os.getenv("COMPANY_COL")
PRODUCT_COL = os.getenv("PRODUCT_COL")
METRICS_CSV = os.getenv("METRICS_CSV")