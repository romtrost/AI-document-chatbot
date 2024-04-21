from RAG.RAG import RAG

if __name__ == "__main__":

    rag = RAG(filename="KTH_Master_Thesis.pdf")
    
    result, sources = rag.run(query="Who is the author?")
    print(result)
    print(sources)