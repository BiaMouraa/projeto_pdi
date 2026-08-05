-- Consulta Top-K por Distancia Euclidiana
-- Operador pgvector: <->
PREPARE semantic_search_euclidean (vector, int) AS
SELECT
    image_id,
    class_name,
    file_path,
    embedding <-> $1 AS score
FROM flower_embeddings
ORDER BY embedding <-> $1
LIMIT $2;

-- Exemplo de uso:
-- EXECUTE semantic_search_euclidean('[0.1,0.2,0.3]'::vector, 5);
-- EXECUTE semantic_search_euclidean('[0.1,0.2,0.3]'::vector, 10);


-- Consulta Top-K por Distancia de Cosseno
-- Operador pgvector: <=>
PREPARE semantic_search_cosine (vector, int) AS
SELECT
    image_id,
    class_name,
    file_path,
    embedding <=> $1 AS score
FROM flower_embeddings
ORDER BY embedding <=> $1
LIMIT $2;

-- Exemplo de uso:
-- EXECUTE semantic_search_cosine('[0.1,0.2,0.3]'::vector, 5);
-- EXECUTE semantic_search_cosine('[0.1,0.2,0.3]'::vector, 10);
