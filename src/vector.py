import pandas as pd
import numpy as np
import networkx as nx
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix
import os
import itertools
import pickle
from gensim.models import Word2Vec

def load_and_prepare_data(file_path):
    """
    Load and prepare data from CSV file
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        print("Original columns:", df.columns.tolist())
        
        # Convert date column if exists
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            
        # Ensure required columns exist
        required_columns = ['DocumentIdentifier', 'Organization', 'E', 'S', 'G']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert ESG columns to boolean
        for col in ['E', 'S', 'G']:
            df[col] = df[col].astype(bool)
            
        print("\nDataFrame Info after preparation:")
        print(df.info())
        print("\nFirst few rows:")
        print(df.head())
        
        return df
        
    except Exception as e:
        print(f"Error in data loading: {str(e)}")
        raise

def filter_non_esg(df):
    """
    Filter rows with at least one ESG flag
    """
    return df[(df['E']==True) | (df['S'] == True) | (df['G'] == True)]

class graph_creator:
    def __init__(self, df):
        self.df = df
        print(f"Initialized graph creator with {len(df)} rows of data")

    def create_graph(self):
        try:
            # Create edges between organizations mentioned in same URL
            print("Creating organization network...")
            df_edge = pd.DataFrame(self.df.groupby("DocumentIdentifier").Organization.apply(list)
                                   ).reset_index()

            # Generate organization pairs
            print("Generating organization pairs...")
            get_tpls = lambda r: (list(itertools.combinations(r, 2)) if
                                  len(r) > 1 else None)
            df_edge["SourceDest"] = df_edge.Organization.apply(get_tpls)
            df_edge = df_edge.explode("SourceDest").dropna(subset=["SourceDest"])

            # Calculate edge weights
            print("Calculating edge weights...")
            source_dest = pd.DataFrame(df_edge.SourceDest.tolist(),
                                       columns=["Source", "Dest"])
            sd_mapping = source_dest.groupby(["Source", "Dest"]).size()
            get_weight = lambda r: sd_mapping[r.Source, r.Dest]
            source_dest["weight"] = source_dest.apply(get_weight, axis=1)

            # Create network graph
            self.organizations = set(source_dest.Source.unique()).union(
                                 set(source_dest.Dest.unique()))
            print(f"Creating graph with {len(self.organizations)} organizations")
            
            self.G = nx.from_pandas_edgelist(source_dest, source="Source",
                target="Dest", edge_attr="weight", create_using=nx.Graph)
            
            print(f"Graph created with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges")
            return self.G
            
        except Exception as e:
            print(f"Error in graph creation: {str(e)}")
            raise

class CustomNode2Vec:
    def __init__(self, dimensions=128, walk_length=80, num_walks=10, workers=4):
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        
    def _generate_walks(self, adj_matrix):
        """Generate random walks"""
        num_nodes = adj_matrix.shape[0]
        walks = []
        for _ in range(self.num_walks):
            for node in range(num_nodes):
                walk = [node]
                for _ in range(self.walk_length - 1):
                    curr = walk[-1]
                    neighbors = adj_matrix[curr].indices
                    if len(neighbors) > 0:
                        walk.append(np.random.choice(neighbors))
                    else:
                        break
                walks.append([str(node) for node in walk])
        return walks

    def fit(self, G):
        """
        Fit Node2Vec model using updated parameters
        """
        try:
            # Convert NetworkX graph to adjacency matrix
            adj_matrix = nx.adjacency_matrix(G)
            adj_matrix = csr_matrix(adj_matrix)
            
            # Generate walks
            print("Generating random walks...")
            walks = self._generate_walks(adj_matrix)
            
            # Train Word2Vec model with updated parameters
            print("Training Word2Vec model...")
            self.model = Word2Vec(
                sentences=walks,
                vector_size=self.dimensions,
                window=5,
                min_count=0,
                sg=1,
                workers=self.workers,
                epochs=self.num_walks
            )
            
            # Store embeddings in a more accessible format
            self.embedding_ = np.zeros((adj_matrix.shape[0], self.dimensions))
            for i in range(adj_matrix.shape[0]):
                self.embedding_[i] = self.model.wv[str(i)]
            
            return self
            
        except Exception as e:
            print(f"Error in Node2Vec fitting: {str(e)}")
            raise

def get_embeddings(G, organizations):
    """
    Generate embeddings using modified Node2Vec and reduce dimensions using PCA
    """
    try:
        print("Generating Node2Vec embeddings...")
        # Use custom Node2Vec implementation
        n2v = CustomNode2Vec(dimensions=128, walk_length=80, num_walks=10)
        n2v.fit(G)
        
        print("Performing PCA dimensionality reduction...")
        embeddings = n2v.embedding_
        pca = PCA(n_components=3)
        principalComponents = pca.fit_transform(embeddings)
        
        d_e = pd.DataFrame(principalComponents, 
                          columns=['Component_1', 'Component_2', 'Component_3'])
        d_e["organization"] = organizations
        
        print(f"Created embeddings with shape: {d_e.shape}")
        return d_e, n2v
        
    except Exception as e:
        print(f"Error in embedding generation: {str(e)}")
        raise

def get_connections(n2v, organizations, topn=25):
    """
    Find similar organizations based on embeddings
    """
    try:
        print(f"Finding top {topn} connections for each organization...")
        embeddings = n2v.embedding_
        
        # Calculate similarities
        similarities = []
        org_to_idx = {org: idx for idx, org in enumerate(organizations)}
        
        for org in organizations:
            org_idx = org_to_idx[org]
            org_embedding = embeddings[org_idx]
            
            # Calculate cosine similarity with all other organizations
            sims = []
            for other_org in organizations:
                if other_org != org:
                    other_idx = org_to_idx[other_org]
                    other_embedding = embeddings[other_idx]
                    sim = np.dot(org_embedding, other_embedding) / (
                        np.linalg.norm(org_embedding) * np.linalg.norm(other_embedding)
                    )
                    sims.append((other_org, sim))
            
            # Sort by similarity and get top N
            sims.sort(key=lambda x: x[1], reverse=True)
            similarities.append(sims[:topn])
        
        # Create DataFrame
        df_sim = pd.DataFrame(index=range(len(organizations)))
        df_sim["organization"] = organizations
        
        for i in range(topn):
            df_sim[f"n{i}_similar_org"] = [s[i][0] for s in similarities]
            df_sim[f"n{i}_similarity"] = [s[i][1] for s in similarities]
        
        print(f"Created connections dataframe with shape: {df_sim.shape}")
        return df_sim
        
    except Exception as e:
        print(f"Error in connection generation: {str(e)}")
        raise

def save_results(G, embeddings_df, connections_df, output_dir):
    """
    Save all results to files
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save graph
        graph_path = os.path.join(output_dir, "organization_graph.pkl")
        with open(graph_path, "wb") as f:
            pickle.dump(G, f)
        print(f"Saved graph to: {graph_path}")
        
        # Save embeddings
        emb_path = os.path.join(output_dir, "organization_embeddings.csv")
        embeddings_df.to_csv(emb_path, index=False)
        print(f"Saved embeddings to: {emb_path}")
        
        # Save connections
        conn_path = os.path.join(output_dir, "organization_connections.csv")
        connections_df.to_csv(conn_path, index=False)
        print(f"Saved connections to: {conn_path}")
        
    except Exception as e:
        print(f"Error in saving results: {str(e)}")
        raise

def process_organization_network(input_csv, output_dir):
    """
    Main function to process organization network data
    """
    try:
        # Load and prepare data
        print("\n1. Loading and preparing data...")
        df = load_and_prepare_data(input_csv)
        df = filter_non_esg(df)
        
        # Create graph
        print("\n2. Creating organization network...")
        creator = graph_creator(df)
        G = creator.create_graph()
        organizations = list(creator.organizations)
        
        # Generate embeddings
        print("\n3. Generating embeddings...")
        embeddings_df, n2v = get_embeddings(G, organizations)
        
        # Find connections
        print("\n4. Finding organization connections...")
        connections_df = get_connections(n2v, organizations)
        
        # Save results
        print("\n5. Saving results...")
        save_results(G, embeddings_df, connections_df, output_dir)
        
        print("\nProcessing completed successfully!")
        return G, embeddings_df, connections_df
        
    except Exception as e:
        print(f"\nError in main processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Configuration
    INPUT_CSV = "/home/rudra-panda/Desktop/error404/company_gdelt_data/top_200_companies_jan10days_2025.csv"
    OUTPUT_DIR = "/home/rudra-panda/Desktop/error404/company_gdelt_data"
    
    try:
        # Process data
        G, embeddings, connections = process_organization_network(INPUT_CSV, OUTPUT_DIR)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Number of organizations: {len(G.nodes())}")
        print(f"Number of connections: {len(G.edges())}")
        print(f"Embedding dimensions: {embeddings.shape}")
        print(f"Number of organization connections: {len(connections)}")
        
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Create Graph and Add Nodes and Edges
# #Create the graph and add the notes 
# organizations = df2.Organization.unique().tolist()
# G = nx.Graph()
# for org in organizations:
#   G.add_node(org)

# COMMAND ----------

# #Get the Edges and add them to the graph
# #df_edge = pd.DataFrame(df2.groupby("URL").Organizations.first())
# df_edge = pd.DataFrame(df2.groupby("URL").Organization.apply(list))
# df_edge = df_edge.reset_index()
# #df_edge.head()

# def get_tuples(row): 
#   if len(row) > 1:
#     return list(itertools.combinations(row,2))
#   else: 
#     return None

# def get_i(row,i): 
#   return row[i]

# df_edge["SourceDest"] = df_edge.Organization.apply(lambda i: get_tuples(i))
# df_edge = df_edge.explode("SourceDest")
# df_edge = df_edge[~df_edge.SourceDest.isnull()]
# df_edge["Source"] = df_edge.SourceDest.apply(lambda i: get_i(i,0))
# df_edge["Dest"] = df_edge.SourceDest.apply(lambda i: get_i(i,1))
# df_edge.head(20)
# df_edge = df_edge[["Source","Dest"]]
# edges = [tuple(r) for r in df_edge.to_numpy()]
# G.add_edges_from(edges)

# COMMAND ----------

# map = df_edge.groupby(['Source', 'Dest']).size()
# map['abbvie']['amgen']
# def get_weight(row,map): 
#   return map[row.Source,row.Dest]
# df_edge["weight"] = df_edge[["Source","Dest"]].apply(lambda i: get_weight(i,map),axis=1)
# df_edge

# COMMAND ----------

# G_test = nx.from_pandas_edgelist(df_edge, 'Source', 'Dest',
#                             create_using=nx.DiGraph(), edge_attr='weight')
# G

# COMMAND ----------

# import pickle
# fp = "/dbfs/mnt/esg/G_1_month.pkl"
# with open(fp, 'wb') as f:
#     pickle.dump(G, f)

# COMMAND ----------

# g2v = NVVV()
# # way faster than other node2vec implementations
# # Graph edge weights are handled automatically
# g2v.fit(G)

# COMMAND ----------

# embeddings = g2v.model.wv.vectors
# embeddings.shape

# COMMAND ----------

# "visa" in em

# COMMAND ----------

# embeddings = g2v.model.wv.vectors
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# principalComponents = pca.fit_transform(embeddings)
# d_e = pd.DataFrame(principalComponents)
# d_e['company'] = organizations
# d_e.to_csv('/dbfs/mnt/esg/10_day_embeddings_pca.csv',index=None)

# COMMAND ----------

# def expand_tuple(row): 
#   return row[0],row[1]

# l = []
# for i in organizations:
#   sim = g2v.model.wv.most_similar(i,topn=25)
#   l.append(sim)
# c = [f"n{i}" for i in range(25)]
# df_sim = pd.DataFrame(l,columns=c)
# df_sim["company"] = organizations
# for i in c: 
#   cols = [i+"_rec",i+"_conf"]
#   df_sim[cols] = df_sim[i].apply(pd.Series)
# df_sim = df_sim.drop(c,axis=1)
# df_sim

# COMMAND ----------

# # Save the data
# spark.conf.set("spark.databricks.delta.properties.defaults.autoOptimize.optimizeWrite", "true")
# spark.conf.set("spark.databricks.delta.properties.defaults.autoOptimize.autoCompact", "true")

# save_path = "dbfs:/mnt/esg/financial_report_data"
# dbutils.fs.mkdirs(save_path)

# file_name = "CONNECTIONS"
# data = spark.createDataFrame(df_sim)
# data.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save(os.path.join(save_path, file_name))
# print(f"Saved to {os.path.join(save_path, file_name)}")

# COMMAND ----------

# save_path = "dbfs:/mnt/esg/financial_report_data"
# file_name = "CONNECTIONS"
# df_sim = load_data(save_path,file_name)
# df_sim.head(10)

# COMMAND ----------

# df_sim.to_csv("/dbfs/mnt/esg/connectionsV2_10days.csv")

# COMMAND ----------

# dbutils.fs.ls("/mnt/esg/")

# COMMAND ----------

# import pickle  # python3

# fp = '/dbfs/mnt/esg/graph.pkl'
# # Dump graph
# with open(fp, 'wb') as f:
#     pickle.dump(G, f)

# COMMAND ----------

# G.edge_subgraph

# COMMAND ----------

# company = 'intercontinental exchange'
# edges = []
# for i in G.edges: 
#   if i[0] == (company): 
#     edges.append(i)
# G2 = G.edge_subgraph(edges)

# COMMAND ----------

# G.nodes

# COMMAND ----------


