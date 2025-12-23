"""
GNN Node Module using Flax, JAX, and Jraph
Implements a Graph Neural Network node processor with message passing
"""
from typing import Callable, Optional, Dict, Any, Sequence

import ray
import jax
import jax.numpy as jnp
from jax import jit
from flax import linen as nn
from flax.core import FrozenDict
import jraph

from core.app_utils import JAX_DEVICE
from gpu_actor.gpu_processor import GPUProcessor
from qf_utils.all_subs import G_FIELDS, ALL_SUBS


class GNNNodeModule(nn.Module):
    """
    Flax Neural Network Module for Graph Neural Network Node Processing.

    This module is a generic, configurable GNN processor. It is configured with
    runnables (e.g., neural network layers) and an index map that specifies
    how to pipe admin_data through the runnables.
    """
    runnables: Dict[str, Sequence[Callable]]
    index_map: Dict[str, Any]
    num_message_passing_steps: int = 3

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """
        Process graph through message passing steps.
        """

        def update_edge_fn(
            edges: jnp.ndarray,
            senders: jnp.ndarray,
            receivers: jnp.ndarray,
            global_features: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Generic edge update function.
            Gathers features, applies runnables, and updates edge features.
            """
            if 'edges' not in self.runnables:
                return edges

            input_features = []
            for source in self.index_map['edges']['inputs']:
                if source == 'edges':
                    input_features.append(edges)
                elif source == 'senders':
                    input_features.append(senders)
                elif source == 'receivers':
                    input_features.append(receivers)
                elif source == 'globals':
                    broadcasted_globals = jnp.broadcast_to(
                        global_features, (edges.shape[0], global_features.shape[-1])
                    )
                    input_features.append(broadcasted_globals)

            concatenated_features = jnp.concatenate(input_features, axis=-1)

            result = concatenated_features
            for runnable in self.runnables['edges']:
                result = runnable(result)

            output_slice = self.index_map['edges']['output_slice']
            if self.index_map['edges'].get('residual', False):
                original_sliced = edges[..., output_slice]
                return edges.at[..., output_slice].set(original_sliced + result)
            else:
                return edges.at[..., output_slice].set(result)

        def update_node_fn(
            nodes: jnp.ndarray,
            sent_messages: jnp.ndarray,
            received_messages: jnp.ndarray,
            global_features: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Generic node update function.
            Gathers features, applies runnables, and updates node features.
            """
            if 'nodes' not in self.runnables:
                return nodes

            input_features = []
            for source in self.index_map['nodes']['inputs']:
                if source == 'nodes':
                    input_features.append(nodes)
                elif source == 'received_messages':
                    input_features.append(received_messages)
                elif source == 'globals':
                    broadcasted_globals = jnp.broadcast_to(
                        global_features, (nodes.shape[0], global_features.shape[-1])
                    )
                    input_features.append(broadcasted_globals)

            concatenated_features = jnp.concatenate(input_features, axis=-1)

            result = concatenated_features
            for runnable in self.runnables['nodes']:
                result = runnable(result)

            output_slice = self.index_map['nodes']['output_slice']
            if self.index_map['nodes'].get('residual', False):
                original_sliced = nodes[..., output_slice]
                return nodes.at[..., output_slice].set(original_sliced + result)
            else:
                return nodes.at[..., output_slice].set(result)

        def update_global_fn(
            aggregated_nodes: jnp.ndarray,
            aggregated_edges: jnp.ndarray,
            global_features: jnp.ndarray
        ) -> jnp.ndarray:
            """
            Generic global update function.
            Gathers features, applies runnables, and updates global features.
            """
            if 'globals' not in self.runnables:
                return global_features

            input_features = []
            for source in self.index_map['globals']['inputs']:
                if source == 'aggregated_nodes':
                    input_features.append(aggregated_nodes)
                elif source == 'aggregated_edges':
                    input_features.append(aggregated_edges)
                elif source == 'globals':
                    input_features.append(global_features)

            concatenated_features = jnp.concatenate(input_features, axis=-1)

            result = concatenated_features
            for runnable in self.runnables['globals']:
                result = runnable(result)

            output_slice = self.index_map['globals']['output_slice']
            if self.index_map['globals'].get('residual', False):
                original_sliced = global_features[..., output_slice]
                return global_features.at[..., output_slice].set(original_sliced + result)
            else:
                return global_features.at[..., output_slice].set(result)

        gn = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
            aggregate_edges_for_nodes_fn=jraph.segment_sum,
            aggregate_nodes_for_globals_fn=jraph.segment_sum,
            aggregate_edges_for_globals_fn=jraph.segment_sum,
        )

        updated_graph = graph
        for _ in range(self.num_message_passing_steps):
            updated_graph = gn(updated_graph)

        return updated_graph


@ray.remote(num_gpus=0, num_cpus=.2)
class Node(GPUProcessor):
    """
    Ray Actor for GNN Node Processing
    
    This actor receives GraphsTuples, processes them through a GNN,
    and sends results to downstream actors.
    """
    
    def __init__(
        self,
        env: str,
        device: str,
        max_index: int,
        amount_nodes: int,
        node_feature_dim: int = 64,
        edge_feature_dim: int = 32,
        global_feature_dim: int = 16,
        hidden_dim: int = 128,
        num_message_passing_steps: int = 3,
        result_actor_name: str = "RESULT_PROCESSOR",
    ):
        """
        Initialize the GNN Node processor
        
        Args:
            env: Environment identifier
            device: Device type (cpu/gpu)
            max_index: Maximum index for processing
            amount_nodes: Number of nodes in the graph
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            global_feature_dim: Dimension of global features
            hidden_dim: Hidden layer dimension
            num_message_passing_steps: Number of message passing iterations
            result_actor_name: Name of actor to send results to
        """
        GPUProcessor.__init__(self)
        
        self.env = env
        self.device = device
        self.amount_nodes = amount_nodes
        self.max_index = max_index
        self.result_actor_name = result_actor_name
        
        # Setup JAX device
        self.gpu = jax.devices(JAX_DEVICE)[0]
        self.test_gpu_ready(device=self.gpu)
        
        # Define runnables and index_map to replicate old behavior
        runnables = {
            'nodes': [
                nn.Dense(hidden_dim),
                nn.relu,
                nn.Dense(node_feature_dim)
            ],
            'edges': [
                nn.Dense(hidden_dim),
                nn.relu,
                nn.Dense(edge_feature_dim)
            ],
            'globals': [
                nn.Dense(hidden_dim),
                nn.relu,
                nn.Dense(global_feature_dim)
            ],
        }

        index_map = {
            'nodes': {
                'inputs': ['nodes', 'received_messages', 'globals'],
                'output_slice': slice(None),
                'residual': True,
            },
            'edges': {
                'inputs': ['edges', 'senders', 'receivers', 'globals'],
                'output_slice': slice(None),
                'residual': False,
            },
            'globals': {
                'inputs': ['aggregated_nodes', 'aggregated_edges', 'globals'],
                'output_slice': slice(None),
                'residual': False,
            }
        }

        # Initialize GNN module
        self.gnn_module = GNNNodeModule(
            runnables=runnables,
            index_map=index_map,
            num_message_passing_steps=num_message_passing_steps,
        )
        
        # Initialize parameters
        self.params = None
        self.initialized = False
        
        # Storage for received graphs
        self.graph_store: Dict[int, jraph.GraphsTuple] = {}
        self.current_index = 0
        
        # JIT compiled functions
        self.process_graph_jit = None
        
        print(f"Node initialized successfully on device: {self.gpu}")
    
    def initialize_params(self, sample_graph: jraph.GraphsTuple):
        """
        Initialize network parameters using a sample graph
        
        Args:
            sample_graph: Sample GraphsTuple for parameter initialization
        """
        if self.initialized:
            return
        
        # Initialize parameters with random key
        rng = jax.random.PRNGKey(0)
        self.params = self.gnn_module.init(rng, sample_graph)
        
        # Create JIT compiled processing function
        @jit
        def process_graph(params, graph):
            return self.gnn_module.apply({'params': params}, graph)
        
        self.process_graph_jit = process_graph
        self.initialized = True
        
        print("GNN parameters initialized successfully")
    
    def receive_graph(
        self,
        graph: jraph.GraphsTuple,
        index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Receive a GraphsTuple for processing
        
        Args:
            graph: GraphsTuple to process
            index: Index of this graph in the sequence
            metadata: Optional metadata about the graph
        
        Returns:
            Success status
        """
        try:
            # Initialize parameters if needed
            if not self.initialized:
                self.initialize_params(graph)
            
            # Store graph
            self.graph_store[index] = graph
            
            print(f"Received graph {index} with {graph.n_node[0]} nodes and {graph.n_edge[0]} edges. Metadata: {metadata}")
            
            return True
        except Exception as e:
            print(f"Error receiving graph: {e}")
            return False
    
    def process_graph(self, index: int) -> Optional[jraph.GraphsTuple]:
        """
        Process a stored graph through the GNN
        
        Args:
            index: Index of graph to process
        
        Returns:
            Processed GraphsTuple or None if error
        """
        try:
            if index not in self.graph_store:
                print(f"Graph {index} not found in store")
                return None
            
            if not self.initialized:
                print("GNN not initialized")
                return None
            
            # Get graph
            graph = self.graph_store[index]
            
            # Process through GNN
            print(f"Processing graph {index}...")
            updated_graph = self.process_graph_jit(self.params, graph)
            
            print(f"Graph {index} processed successfully")
            
            return updated_graph
            
        except Exception as e:
            print(f"Error processing graph {index}: {e}")
            return None
    
    def process_and_send(self, index: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Process graph and send to result actor
        
        Args:
            index: Index of graph to process
            metadata: Optional metadata to send with results
        
        Returns:
            Success status
        """
        try:
            # Process graph
            updated_graph = self.process_graph(index)
            
            if updated_graph is None:
                return False
            
            # Get result actor
            try:
                result_actor = ray.get_actor(self.result_actor_name)
            except ValueError:
                print(f"Result actor '{self.result_actor_name}' not found")
                return False
            
            # Send results
            print(f"Sending processed graph {index} to {self.result_actor_name}...")
            ray.get(result_actor.receive_processed_graph.remote(
                graph=updated_graph,
                index=index,
                metadata=metadata or {}
            ))
            
            # Clean up processed graph from store
            del self.graph_store[index]
            
            print(f"Graph {index} sent successfully")
            
            return True
            
        except Exception as e:
            print(f"Error in process_and_send for graph {index}: {e}")
            return False
    
    def process_batch(self, indices: list[int]) -> bool:
        """
        Process multiple graphs in sequence
        
        Args:
            indices: List of graph indices to process
        
        Returns:
            Success status
        """
        try:
            for idx in indices:
                success = self.process_and_send(idx)
                if not success:
                    print(f"Failed to process graph {idx}")
                    return False
            
            print(f"Batch processing completed for {len(indices)} graphs")
            return True
            
        except Exception as e:
            print(f"Error in batch processing: {e}")
            return False
    
    def update_params(self, new_params: FrozenDict):
        """
        Update network parameters (for training)
        
        Args:
            new_params: New parameter dictionary
        """
        self.params = new_params
        print("Parameters updated")
    
    def get_params(self) -> Optional[FrozenDict]:
        """
        Get current network parameters
        
        Returns:
            Current parameters or None
        """
        return self.params
    
    def get_index(self) -> int:
        """Get current processing index"""
        return self.current_index
    
    def set_index(self, index: int):
        """Set current processing index"""
        self.current_index = index
    
    @staticmethod
    def get_parent(ntype: str) -> str:
        """Get parent type for a node type"""
        if ntype in G_FIELDS:
            return "GAUGE"
        if ntype in ALL_SUBS:
            return "FERMION"
        return "HIGGS"
    
    def clear_store(self):
        """Clear all stored graphs"""
        self.graph_store.clear()
        print("Graph store cleared")
    
    def get_store_size(self) -> int:
        """Get number of graphs in store"""
        return len(self.graph_store)
