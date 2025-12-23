"""
Improve and run the followign prompt:
Create a spanner serve.deployment from the
given python class. include fastapi routes
to:
- /create instance, db, table,graph, change_stream (include in each route a check igf the resource already exists)
- /upsert to table
- /read-change-stream
- /get-neighbors ( from a given G(raph) )
- /delete-instance ( and all rcs inisde )

Extras:
- include  prints with emojicons
- run on cpu
- for each incomming request use the spanner async client and create a asyncio.tasks to process multille requeste parallel
- include a sync method in the init class of the actor to create all resources

"""

import os
GCP_ID=os.environ.get("GCP_ID")
GCP_REGION=os.environ.get("GCP_REGION")

SP_INSTANCE_ID=os.environ.get("SP_INSTANCE_ID")
SP_DB_ID=os.environ.get("SP_DB_ID")


