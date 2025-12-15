import asyncio

from rest_framework.response import Response
from rest_framework.views import APIView


def process_configuration(config_data):
    # This is your predefined function that processes the configuration
    # For demonstration, we'll just print the data
    print("Processing configuration:", config_data)
    # Add your actual processing logic here
    return "Configuration processed successfully"


class ConfigurationView(APIView):
    def post(self, request):
        graph_model = GraphHandler()
        if ACTION["graph"]["create"]:
            print("Working sne's")
            asyncio.run(graph_model.main())

        if ACTION["model"]["train"]:
            if ACTION["model"]["type"] == "sage":
                print("Train")
                sage_trainer = GraphSAGETrainer(
                    graph_model.G,
                    info=SRC_PATH,
                    embedding_dim=1024,
                    hidden_dim=32,
                    num_layers=8,
                    sample_neighbors=5,
                    epochs=5,
                    layers=graph_model.success_list
                )

                # Step 3: Train the model
                sage_trainer.train()
            else:
                gtm = HeteroGraphConvGTTrainer(bucket_name="bestbrain",
                                               info=SRC_PATH)  # GraphTransformerModel(g_utils=graph_model.g_utils)

                asyncio.run(gtm.main())

        if ACTION["model"]["test"]:
            if ACTION["model"]["type"] == "sage":
                trainer = GraphSAGETrainer(graph_model.G, info=SRC_PATH, embedding_dim=1024, hidden_dim=32,
                                           num_layers=8,
                                           sample_neighbors=5, epochs=5, layers=graph_model.success_list)
                trainer.load_model()

                # Initialize visualizer AFTER model is loaded
                visualizer = GraphSAGEVisualizer(trainer)

                # Visualize the entire graph structure
                visualizer.visualize_graph()

                # Visualize a sampled neighborhood around a specific node
                visualizer.visualize_sampled_graph(node_id="GO:0099593", num_hops=2)

                # Visualize trained GraphSAGE embeddings in 2D
                visualizer.visualize_embeddings()
        if ACTION["graph"]["merge"]:
            print("Handling Merge")
            print("--- ALERT --- MERGE SHOULD BE CALLED OUTSIDE OF THE PIPE (SINGLE)")
            for k, v in graph_model.files.items():
                asyncio.run(graph_model.utils.check_ckpt(k))
                if not v.get("embed"):
                    asyncio.run(graph_model.embeds.main())

                elif not v.get("sne"):
                    asyncio.run(graph_model.process_layers(v, k))

        if ACTION["rag"]["create"]:
            print("hi")
            asyncio.run(graph_model.index_from_graph(
                key="eco"
            ))
        if ACTION["rag"]["test"]:
            #asyncio.run(query_graph_vectorstore("Find molecules related to metabolism", top_k=30))
            pass
        return Response()



