from backend.utils.vector_store import PostgresVectorStore

vs = PostgresVectorStore()
res = vs.semantic_search_customers("Find customers similar to 'CUST010'.", limit=5, dataset_id='default')
print(res)
