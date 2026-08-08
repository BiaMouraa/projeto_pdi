-- Consultas Top-K para o dataset Oxford Flower 17 (tabela legada).
-- Operador pgvector Euclidiana: <->
PREPARE semantic_search_euclidean_oxford17 (vector, int) AS
SELECT
    image_id,
    class_name,
    file_path,
    embedding <-> $1 AS score
FROM flower_embeddings
ORDER BY embedding <-> $1
LIMIT $2;

-- Operador pgvector Cosseno: <=>
PREPARE semantic_search_cosine_oxford17 (vector, int) AS
SELECT
    image_id,
    class_name,
    file_path,
    embedding <=> $1 AS score
FROM flower_embeddings
ORDER BY embedding <=> $1
LIMIT $2;


-- Consultas Top-K para o dataset Oxford Flower 102 (tabela nova).
PREPARE semantic_search_euclidean_oxford102 (vector, int) AS
SELECT
    image_id,
    class_name,
    file_path,
    embedding <-> $1 AS score
FROM flower_embeddings_oxford102
ORDER BY embedding <-> $1
LIMIT $2;

PREPARE semantic_search_cosine_oxford102 (vector, int) AS
SELECT
    image_id,
    class_name,
    file_path,
    embedding <=> $1 AS score
FROM flower_embeddings_oxford102
ORDER BY embedding <=> $1
LIMIT $2;

-- Exemplos:
-- EXECUTE semantic_search_cosine_oxford17('[0.1,0.2,...]'::vector, 5);
-- EXECUTE semantic_search_cosine_oxford102('[0.1,0.2,...]'::vector, 10);
