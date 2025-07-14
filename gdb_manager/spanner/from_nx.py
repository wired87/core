import asyncio

import networkx as nx

from utils.file import aread_json_content
from ggoogle.spanner import create_default_node_table_query
from ggoogle.spanner.graph_loader import SpannerGraphLoader
from google.cloud.spanner_v1 import param_types


class SpannerFromNx(SpannerGraphLoader):
    def __init__(self):
        super().__init__()
        self.embed_dim = "1d"

    def load_graph(self, path):
        """Loads a NetworkX graph from a JSON file asynchronously."""
        content = asyncio.run(aread_json_content(path))
        G = nx.node_link_graph(content)
        return G

    def extr_schema(self, G):
        """Extracts schema from the NetworkX graph and ensures uniqueness."""
        print("START extr_schema ")
        nodes, edges = {}, {}
        # pre fetch all table schemas
        for src, attrs in G.nodes(data=True):
            t = attrs.get('type').upper()  # none
            # print("Working src", src, t)
            if t not in nodes:
                nodes[t] = {}
                nodes[t]["schema"] = self.get_table_columns(t)

            if "id" not in nodes[t]["schema"]:
                nodes[t]["schema"]["id"] = src

            self.unique_schema(attrs, nodes, t)

        for src, trgt, attrs in G.edges(data=True):
            src_layer = attrs.get("src_layer").upper()
            trgt_layer = attrs.get("trgt_layer").upper()
            relationship = attrs.get("rel").lower()
            t = f"{src_layer}_{relationship}_{trgt_layer}"
            if t not in edges:
                edges[t] = {}
            if "schema" not in edges[t]:
                edges[t]["schema"] = self.get_table_columns(t)

            if "id" not in edges[t]["schema"]:
                edges[t]["schema"]["id"] = "STRING(MAX)"

            self.unique_schema(attrs, edges, t)

        print("NODE SCHEMAS SET:", [f"{k}:{len(v['schema'])}" for k, v in nodes.items()])
        print("EDGE SCHEMAS SET:", [f"{k}:{len(v['schema'])}\n" for k, v in edges.items()])
        return nodes, edges

    def split_graph(self, G):
        """Splits the graph into node and edge data with batch preparation."""
        nodes, edges = self.extr_schema(G)
        for src, attrs in G.nodes(data=True):
            t = attrs.get('type').upper()
            # print("node type", t)
            if "rows" not in nodes[t]:
                nodes[t]["rows"] = []
            values_dict = dict(id=src, **{k: v for k, v in attrs.items() if k not in ["id"]})
            nodes[t]["rows"].append(
                self.convert_row_uniform_format(
                    schema=nodes[t]["schema"],
                    values_dict=values_dict
                )
            )

        # evtl edge erstellung fehler
        for src, trgt, attrs in G.edges(data=True):
            try:
                # print(src, trgt)
                src_layer = attrs.get("src_layer", "default").upper()
                trgt_layer = attrs.get("trgt_layer", "default").upper()
                relationship = attrs.get("rel").lower()
                t = f"{src_layer}_{relationship}_{trgt_layer}"

                #print("edge attrs", attrs)

                if "rows" not in edges[t]:
                    #print("creating rows for ", t)
                    edges[t]["rows"] = []

                values_dict = dict(id=self.edge_id(attrs, src, trgt), src=src, trgt=trgt, **{k: v for k, v in attrs.items()})
                edges[t]["rows"].append(values_dict)
            except Exception as e:
                print(f"Error during edge creation: {e}")

        print("NODE ROWS SET:", [f"{k}:{len(v['rows'])}" for k, v in nodes.items()])
        print("EDGE ROWS SET:", [f"{k}:{len(v['rows'])}" for k, v in edges.items()])

        # save memory
        G.remove_edges_from(list(G.edges()))
        G.remove_nodes_from(list(G.nodes()))
        print("Rm all nodes and edges from nx")

        return nodes, edges


    def from_nx(self, G: nx.Graph):
        # Collect nodes and edges in separate spaces
        # 1 receive for EACH TABLE: schema, rows to insert

        nodes, edges = self.split_graph(G)

        for table_name, v in nodes.items():
            # check table, append schema
            if not self.spanner_table_exists(table_name):
                self.create_table(
                    query=create_default_node_table_query(table_name=table_name)
                )

            self.add_columns_bulk(table_name=table_name, attrs=v["schema"])
            #print("Reference item:", v["rows"][len(v["rows"])-1])

            self.batch_process_rows(
                table_name=table_name,
                id_column_name="id",
                rows=v["rows"],
            )

        for table_name, v in edges.items():
            if not self.spanner_table_exists(table_name):
                edge_item = v["rows"][0]
                src_layer = edge_item.get("src_layer")
                trgt_layer = edge_item.get("trgt_layer")
                if src_layer:
                    src_layer = src_layer.upper()
                else:
                    print("No src_layer for ", edge_item)
                if trgt_layer:
                    trgt_layer = trgt_layer.upper()
                else:
                    print("No trgt_layer for ", edge_item)

                self.create_edge_table_batch(
                    srcl=f"{src_layer}",
                    trgtl=f"{trgt_layer}",
                    t=table_name,
                )

            self.add_columns_bulk(
                table_name=table_name,
                attrs=v["schema"]
            )

            self.batch_process_rows(
                table_name=table_name,
                id_column_name="id",
                rows=v["rows"],
            )


    def unique_schema(self, attrs, item, t):
        for key, value in attrs.items():

            if key not in item[t]["schema"]:
                if key == "range":
                    key = f"c{key}"
                item[t]["schema"][key] = self.get_spanner_type(value)


        #print("Schema set", item[t]["schema"])


    def edge_id(self, attrs, src, trgt):
        relationship = attrs.get("rel").lower()

        edge_type = attrs.get("type")
        if not edge_type:
            edge_type = relationship
        edge_id = f"{src}_{edge_type}_{trgt}"
        return edge_id

    def gather_existing_ids(self, nodes, edges):
        print("START gather_existing_ids")
        for k in nodes.keys():
            if "ids" not in nodes[k]:
                nodes[k]["ids"] =[]
            nodes[k]["ids"].extend(self.get_table_ids(table_name=k, id_column_name="id"))

        for k in edges.keys():
            if "ids" not in edges[k]:
                edges[k]["ids"] =[]
            edges[k]["ids"].extend(self.get_table_ids(table_name=k, id_column_name="id"))
        print("FINSIHED gather_existing_ids")
        return nodes, edges

    def convert_row_uniform_format(self, schema: dict, values_dict: dict):
        """
        schema: dict -> col_name : type
        Bring all nodes/edge values in the same order as schema and return as a tuple.
        Ensures that no duplicate entries are included.
        """
        # print("Extracted Schema", schema)
        new_values = {}
        for k, v in schema.items():
            if k not in new_values:
                existing_entry = values_dict.get(k)
                if existing_entry:
                    new_values[k] = existing_entry
                else:
                    new_values[k] = None
        # print("new_values", new_values)
        return new_values



    def generate_embeddings_for_chunk(self, rows, embedding_model_name, schema, separator):
        """
        Generates embeddings for a chunk of rows.
        """
        # Extract values from each row and concatenate them
        concatenated_values = []
        for row in rows:
            row_values = [str(row[col]) for col in schema if col != "embedding"]
            concatenated_values.append(separator.join(row_values))

        # Generate embeddings using the embedding model
        with self.database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                f"SELECT embeddings.values FROM ML.PREDICT(MODEL @model, (SELECT @contents AS content))",
                params={"model": embedding_model_name, "contents": concatenated_values},
                param_types={"model": param_types.STRING, "contents": param_types.ARRAY(param_types.STRING)},
            )
            embeddings = [row[0] for row in results]
        return embeddings