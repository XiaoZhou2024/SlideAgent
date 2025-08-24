import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text, Engine


class DatabaseManager:
    """
    数据库管理器，负责连接数据库和执行SQL查询
    """

    def __init__(
            self,
            db_user: str = 'ikun',
            db_password: str = 'wwwhelloworld111',
            db_host: str = 'frp.bnuzh.top',
            db_port: str = '14532',
            db_name: str = 'RealEstate',
    ):
        self.db_user = db_user
        self.db_password = db_password
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name

        self.engine: Engine = self._create_engine()

    def _create_engine(self) -> Engine:
        url = (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
        engine = create_engine(url)
        return engine

    def execute_query_save_data(self, sql_query: list[str], data_path: Path):
        base_path = Path(data_path)
        retrieval_path = base_path / "retrieval"
        retrieval_path.mkdir(parents=True, exist_ok=True)
        with self.engine.connect() as conn:
            for i, query in enumerate(sql_query):
                block_data = pd.read_sql(query, conn)
                csv_name = '{}.csv'.format(i)
                script = os.path.join(retrieval_path, csv_name)
                block_data.to_csv(script, index=False, encoding='utf-8-sig')


