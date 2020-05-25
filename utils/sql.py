import os
import sqlite3
from sqlite3 import Error

from utils.constants import SQL_DATABASE_PATH, TEST_SQL_DATABASE_PATH


def create_connection(db_file):
    """
    Create a database connection to the SQLite database specified by db_file

    Args:
        db_file: database file path

    Returns: Connection object or None

    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn


def create_table(conn, create_table_sql):
    """
    Create a table from the create_table_sql statement

    Args:
        conn: Connection object
        create_table_sql: sql statement to create table

    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)


def main():
    sql_create_projects_table = """ 
    CREATE TABLE IF NOT EXISTS projects (
    id integer PRIMARY KEY,
    name text NOT NULL,
    begin_date text,
    end_date text
    ); 
    """
    conn = create_connection(TEST_SQL_DATABASE_PATH)
    if conn is not None:
        create_table(conn, sql_create_projects_table)
        print("Project table has been created")
    else:
        print("Cannot create database connection")


if __name__ == "__main__":
    main()
