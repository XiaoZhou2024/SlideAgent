import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, Engine

load_dotenv()

class DatabaseManager:
    """
    Executes SQL queries and saves the results to CSV files.

    Args:
        sql_query (list): A list of SQL query strings to execute
        data_path (Path): The directory path where results will be saved as CSV files

    The method creates a 'retrieval' subdirectory in the specified path and saves
    each query result as a separate CSV file named sequentially as '0.csv', '1.csv', etc.
    """
    def __init__(
            self,
            db_user: str = None,
            db_password: str = None,
            db_host: str = None,
            db_port: str = None,
            db_name: str = None,
    ):
        self.db_user = db_user or os.getenv('DB_USER')
        self.db_password = db_password or os.getenv('DB_PASSWORD')
        self.db_host = db_host or os.getenv('DB_HOST')
        self.db_port = db_port or os.getenv('DB_PORT')
        self.db_name = db_name or os.getenv('DB_NAME')

        self.engine: Engine = self._create_engine()

    def _create_engine(self) -> Engine:
        url = (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
        engine = create_engine(url)
        return engine

    def execute_query_save_data(self, sql_query: list, data_path: Path):
        base_path = Path(data_path)
        retrieval_path = base_path / "retrieval"
        retrieval_path.mkdir(parents=True, exist_ok=True)
        with self.engine.connect() as conn:
            for i, query in enumerate(sql_query):
                if len(query) == 1:
                    block_data = pd.read_sql(query[0], conn)
                    csv_name = '{}.csv'.format(i)
                    script = os.path.join(retrieval_path, csv_name)
                    block_data.to_csv(script, index=False, encoding='utf-8-sig')
                    print(f"Retrieval file written to {script}")




