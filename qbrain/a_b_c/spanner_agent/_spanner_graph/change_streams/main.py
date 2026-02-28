from google.cloud.spanner_v1 import ExecuteSqlRequest

from qbrain.a_b_c.spanner_agent._spanner_graph.acore import ASpannerManager

from google.cloud.spanner_v1.services import spanner

class ChangeStreamMaster(ASpannerManager):
    """
    A class to listen to Spanner change streams on a 'Nodes' table
    and notify a 'center' node when an attribute of one of its 'neighbors' changes.
    """

    def __init__(self):
        """
        Initializes the listener with Spanner client and table/stream names.
        """
        super().__init__()

    def create_change_stream_query(
            self,
            change_stream_name,
            table_names:list,
    ):
        query = f"CREATE CHANGE STREAM {change_stream_name}\n"
        #for table in self.nodes_tables:
        tables_clause = ", ".join(table_names)
        query += f"\n  [FOR {tables_clause}]"
        options = f"""
        \n
        [
            OPTIONS (
                retention_period = '7D',
                value_capture_type = NEW_ROW,
                exclude_ttl_deletes = true,
                exclude_insert = true,
                exclude_update = false,
                exclude_delete = true,
                allow_txn_exclusion = true
            )
        ]
        """
        query += options
        return query


    def create_change_stream(self, name, table_names):
        query = self.create_change_stream_query(
            table_names=table_names,
            change_stream_name=name,
        )
        try:
             response = self.database.update_ddl([query])
             print(f"Change stream created: {response}")
             return True
        except Exception as e:
            print(f"Err creating change stream: {e}")
        return False


    def read_change_stream_query(
            self,
            stream_name: str,
    ) -> str:
        change_stream_query = f"""
        SELECT
            *
        FROM
            READ_{stream_name}(
                start_timestamp => @start_time,
                end_timestamp => @end_time,  -- Read up to the current time/infinitely
                partition_token => NULL,
                heartbeat_milliseconds => 5000 -- Send a heartbeat every 5 seconds
            )
        """
        return change_stream_query


    async def read_change_stream(self, stream_name, start_time, end_time):
        query = self.read_change_stream_query(
            stream_name=stream_name
        )
        request = ExecuteSqlRequest(
            session=self.session.name,
            sql=query,
            params={
                "start_time": start_time,
                "end_time": end_time,
            },
            param_types={
                "start_time": spanner.param_types.Array(spanner.param_types.STRING),
                "end_time": spanner.param_types.Array(spanner.param_types.STRING),
            },
        )
        result = await self.aclient.execute_sql(
            request=request,
            timeout=9.0
        )
        # 5. Process the stream of results
        final_results = []
        for row in result:
            record_json = row[0][0]
            final_results.append(record_json)
        return final_results

