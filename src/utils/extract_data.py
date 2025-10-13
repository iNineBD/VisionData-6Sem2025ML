import pandas as pd
import sqlalchemy
import os
from urllib.parse import quote_plus

username = os.getenv("SQLSERVER_USERNAME")
password = os.getenv("SQLSERVER_PASSWORD")
host = os.getenv("SQLSERVER_HOST")
port = os.getenv("SQLSERVER_PORT")
database = os.getenv("SQLSERVER_DATABASE")


def get_data(query_path: str) -> pd.DataFrame:
    """Extra dados do SQL Server através de uma query SQL.
    Args:
        query_path (str): Caminho para o arquivo SQL contendo a query.
    Returns:
        pd.DataFrame: DataFrame com os dados extraídos.
    """

    connection = (
        f"mssql+pyodbc://{username}:{quote_plus(password)}@{host}:{port}/{database}"
        f"?driver=ODBC+Driver+17+for+SQL+Server"
    )

    engine = sqlalchemy.create_engine(connection)

    with open(query_path, "r", encoding="utf-8") as file:
        query = file.read()

    df = pd.read_sql_query(query, con=engine)

    return df
