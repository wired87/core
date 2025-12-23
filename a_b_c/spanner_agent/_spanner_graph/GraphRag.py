import asyncio
import pprint
from typing import List

from bm.settings import TEST_USER_ID
from GenomeDataProcessor.extractors.functions.goterm import RELATION_TYPES
from GenomeDataProcessor.extractors.functions.goterm.gocore import GoCore
from GenomeDataProcessor.extractors.functions.goterm.urls import genes_for_goterm_url
from utils import gene_info_url
from _google.spanner import create_goterm_rag_result_table_query, create_default_table_query
from _google.spanner.acore import ASpannerManager
from _google.spanner.dj import SpannerEmbedder
from _google.spanner.graph_loader import SpannerRAG
from _google.vertexai import GEM
from _google.vertexai.gem.query_to_keywords import gem_extract_goterm_keywords
from _google.generativeai.types.content_types import Tool, FunctionDeclaration  # Import the correct types.

from utils import embed
from utils.utils import Utils


class GotermRagProcess(SpannerEmbedder, ASpannerManager, SpannerRAG):
    def __init__(self, query, functionalities: List or None, user_id, job_id, term_limit: int = 25):
        """
        :param query: Additional Prompt
        :param functionalities: [Dendrite, Nucleus,...]
        Todo: Real time Graph (fetched admin_data get converted to graph, embedded, given to llm)
        ToDo: Pack Ensembl in BQ for own server (-> ens rate limit (15/s) tripples the time/request -> einfach endlich ens graph repr. hochladen!!!
        ToDo: Save each generated keyword and further results in the GotermRagProcess-Spanner, Table -> minim. time and llm cost
        ToDO: fetch descr, from go api and embed again
        """
        super().__init__()
        self.utils = Utils()
        self.query = query
        self.functionalities = functionalities
        self.term_limit = term_limit
        self.user_id = user_id
        self.job_id = job_id
        self.gocore = GoCore()
        self.go_table_name = "RAG_RESULTS"
        self.testing = True
        self.keyword_gen_tool = Tool(
            function_declarations=[
                FunctionDeclaration(
                    name="generate_keywords_for_goterm_vectorsearch",
                    description="Generate a list of 20 search keywords from a given functional biological term to "
                                "extract all Goterms using a vector search from the Gene Ontology Database.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "functional_term": {
                                "type": "string",
                                "description": "A biological function term to generate keywords for GO term retrieval"
                            }
                        },
                        "required": ["functional_term"]
                    },

                )
            ]
        )
        self.chat = GEM.start_chat()

    async def function_to_term(self, testing=False):
        """
        - Check cache (spanner entry for functionalitie) -> later
        - Create keywords for goterm (for each given functionaity)
        - GoT API fetch Genes _> uniprot -> local ens gene extraction
        :return:
        """
        """if testing:
            struct = {}
            struct=[{'term': 'membrane', 'answer': 'plasma membrane, cell membrane, outer membrane, inner membrane, nuclear membrane, mitochondrial membrane, endoplasmic reticulum membrane, Golgi membrane, lysosomal membrane, vacuolar membrane, bacterial membrane, archaeal membrane, membrane protein, transmembrane protein, integral membrane protein, peripheral membrane protein, lipid bilayer, membrane fluidity, membrane transport, membrane trafficking\n', 'go_terms': [{"id":'GO:0000139', "distance":0.3096607349756014}]}]
            struct = await self.get_genes_from_gt(struct)
            #pprint.pp(struct)
            struct = await self.ens_details(struct)
            #pprint.pp(struct)
            return"""

        # check for term row in resultdb
        struct = []

        # cache_struct = await self.check_cache()

        if len(self.functionalities):
            # create keywords
            struct = await self.create_keywords_goterm()
            # -> check spanner for similar keywords
            """
            Example keywords:
            [{term: dendrite, keywords: [all, key, words]}]
            """
            # perfrom similarity search spanner
            struct = await self.vector_search(struct)
            """
            Example response:
            [{term: dendrite, keywords: [all, key, words], go_terms: [{id:1, distance: .78]}]
            """
            # gather genes

        """if len(cache_struct) and struct:
            struct.extend(cache_struct)
        elif not len(struct):
            struct = cache_struct"""

        struct = await self.descendent_subgraph(struct)

        struct = await self.get_genes_from_gt(struct)

        print("RAG job finished")
        #pprint.pp(struct)
        return struct
    # everythign is jist intelligent energy that uses matter in "trad" sense
    async def descendent_subgraph(self, struct):
        async def descendent_processor(entry):
            await asyncio.gather(*[
                self.gocore.subgraph(go["id"], descendent_only=True, add_to_dict=go)
                for go in entry["go_terms"]
            ])
            return entry

        updated_struct = await asyncio.gather(
            *[
                descendent_processor(item)
                for item in struct
            ]
        )
        print("UPDATED STRUCT:")
        #pprint.pp(updated_struct)

        return updated_struct


    async def get_genes_from_gt(self, struct: list[dict]) -> list[dict]:
        """
        Given a list of structures with GO terms, fetch associated genes and return enriched admin_data.

        Each `struct` item should look like:
        {
            "term": str,
            "keywords": list[str],
            "go_terms": [
                {
                    "id": str,
                    "distance": float,
                    ...
                }
            ]
        }
        """

        async def term_type_processor(go_rel_type: str, gid: str) -> dict:
            genes = await self.utils.aget(url=genes_for_goterm_url(gid, go_rel_type))
            print("Genes extracted", type(genes))
            if isinstance(genes, tuple):
                genes = genes[0]
            if genes is not None:
                return {
                    go_rel_type: {
                        "genes": [
                            row.get("subject", {}).get("label")
                            for row in genes.get("associations")
                        ]
                    }
                } if genes.get("associations") is not None else {}


        async def gene_gather_processor(entry: dict) -> None:
            print("gene_gather_processor", entry)
            gid = entry.get("id")
            if gid:

                await asyncio.gather(*[
                    gene_gather_processor(descendent)
                    for descendent in entry.get("descendents", [])
                ])

                collected = {}
                collected.update(await asyncio.gather(*[
                    term_type_processor(go_rel_type, gid) for go_rel_type in RELATION_TYPES
                ]))
                entry["genes"] = {}
                entry["genes"].update(collected)

        async def handle_gene_name_extraction(item: dict) -> dict:
            await asyncio.gather(
                *[
                     gene_gather_processor(entry)
                     for entry in item.get("go_terms", [])
                 ] + [
                     gene_gather_processor(entry["descendents"])
                     for entry in item.get("go_terms", []) if len(entry["descendents"])
                 ])
            return item

        return await asyncio.gather(*[
            handle_gene_name_extraction(item) for item in struct
        ])



    async def check_cache(self):
        struct = []
        keys_to_rm = []

        async def func_processor(term):
            result: list = self.spanner_vector_search(
                data=embed(term),
                table_name="RAG_RESULTS",
                custom=True,
                limit=1,
                select=[
                    "keywords",
                    f"go_terms_{self.term_limit}",
                    "term",
                    "id"
                ],
                embed_row="id"
            )
            print("result cache req", result)
            if result and isinstance(result[-1], (int, float)) and result[-1] >= .98:
                struct.append({
                    "keywords": result[0],
                    "go_terms": [{"id": term} for term in result[1]],
                    "term": result[2],
                    "cache_distance": result[-1]
                })
                keys_to_rm.append(term)

                print("Add user to cache entry")
                await self.update_list(
                    table=self.go_table_name,
                    col_name="users",
                    new_values=[self.user_id],
                    id_insert=result[3]
                )

        await asyncio.gather(*[func_processor(term) for term in self.functionalities])

        for item in keys_to_rm:
            self.functionalities.remove(item)
        return struct

    def save_results(self, stuff, user_id, job_id, query, term_limit):
        # Save results
        schema = self.check_add_table(self.go_table_name, ttype=create_default_table_query(self.go_table_name))
        # todo create cols dynamic -> term_limit = perfect reusable
        # Extend upsert object
        final_rows = []
        for i, item in enumerate(stuff):
            row = {}
            row[
                "id"] = f"{item.get('term').lower().replace(' ', '_').replace('-', '_')}_{self.user_id}_{self.job_id}_{i}"
            row["term"] = item.get("term")
            row["users"] = [str(user_id)]
            row[f"go_terms_{item.get('term_limit')}"] = [term["id"] for term in item["go_terms"]]
            row["keywords"] = item["answer"] if isinstance(item["answer"], list) else item["answer"].split(",")
            row["term_limit"] = term_limit
            final_rows.append(row)
        # embeds of keywords, user_ids:list,
        print("final_rows 0", final_rows[0])
        self.update_insert(self.go_table_name, stuff, schema, insert_only=True)
        print("Process finished")

    async def ens_details(self, struct: list):
        # Run all entries in parallel
        results = await asyncio.gather(*[self.get_filter_info(entry) for entry in struct])
        print("Finished gene processing")
        return results

    async def get_filter_info(self, entry):
        """
        Fetch Ensembl gene details and update struct with correct assignments.
        """
        print("entry", entry)
        tasks = []
        for gt in entry.get("go_terms", []):
            for gene in gt.get("genes", []):
                if self.testing:
                    print("fetch fom spanner")
                    query = self.custom_entries_query(
                        table_name="GENE",
                        check_key="name",
                        check_key_value=gene["name"],
                        select_table_keys=["id"]
                    )
                    await self.asnap(query)
                    tasks.append(self.fetch_gene_info(gene))

        # Run all requests in parallel
        await asyncio.gather(*tasks)
        return entry

    async def fetch_gene_info(self, gene):
        """
        Fetch gene details from Ensembl API and update the gene dictionary.
        """
        try:
            url = gene_info_url(gene.get("display_name"))
            print("Request URL:", url)
            response = await self.utils.aget(url=url)

            # Ensure response contains necessary fields before assignment
            gene.update({
                "id": response.get("id"),
                "species": response.get("species"),
                "object_type": response.get("object_type"),
                "source": response.get("source"),
                "biotype": response.get("biotype"),
                "seq_region_name": response.get("seq_region_name"),
                "end": response.get("end"),
                "start": response.get("start"),
                "db_type": response.get("db_type"),
                "transcripts": len(response.get("Transcript", [])),  # Ensure it's a list
                "description": response.get("description"),
                "display_name": response.get("display_name"),
            })
        except Exception as e:
            await asyncio.sleep(1)
            print("Exception while fetching gene details:", e)
            await self.fetch_gene_info(gene)


    ################### GEM GO
    async def create_keywords_goterm(self):
        print("Create keywords")
        results: list = await asyncio.gather(*[gem_extract_goterm_keywords(term) for term in self.functionalities])
        print("All Keywords", results)
        return results



    async def vector_search(self, struct):
        for item in struct:  # {"term": term, "answer": answer.candidates[0].content.parts[0]}
            item["go_terms"]: list = self.spanner_vector_search(item["answer"])
        print("Spanner entries extracted", struct, "\n\n")
        return struct



    def save_results(self, struct, job_id):
        if not self.spanner_table_exists(self.go_table_name):
            self.create_table(
                query=create_goterm_rag_result_table_query()
            )
        for item in struct:
            # loop through each functionality
            final_row = {
                "id": f"{self.user_id}_{job_id}_{item['term']}",  # later user_id + job_id + term
                "job_id": job_id,
                "term": item['term'],
                "keywords": item['keywords'],
                "go_terms": [term["id"] for term in item['go_terms']],
                "genes": [item["display_name"] for item in item['genes']],
                "term_limit": self.term_limit,
            }
            self.update_insert(table=self.go_table_name, rows=[final_row])


"""

Nucleus

Cytoplasm

Plasma membrane

Mitochondrion

Endoplasmic reticulum (ER)

Golgi apparatus

Ribosome

Lysosome

Peroxisome

Cytoskeleton

Centrosome

Nucleolus

Chromatin

Vesicles

Cilia / Flagella

"""
if __name__ == "__main__":
    try:
        s = GotermRagProcess(
            functionalities=[
                "Cytochrome P450 Enzyme Complex",
                "Bacterial Microcompartments (BMCs)",
                "ATP Synthase Complex"
            ],
            user_id=TEST_USER_ID,
            job_id=str(5),
            term_limit=10,
            query=""
        )
        asyncio.run(s.function_to_term())
    except Exception as e:
        print("Exception during run:", e)
    finally:
        import grpc

        try:
            grpc.aio._exit()
        except Exception:
            pass
