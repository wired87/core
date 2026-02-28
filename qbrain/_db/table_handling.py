def db_table_exists(con, table_name: str) -> bool:
    result = con.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = ?
    """, [table_name]).fetchone()

    return result[0] > 0