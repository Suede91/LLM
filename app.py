from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
import chainlit as cl
from huggingface_hub import hf_hub_download
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
VECTORSTORE_CHROMADB_PATH = 'vectorstore/chromadb'
MODELS_PATH = 'models'

def factory():

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. 
    Preserve the ogirinal question in the answer setiment during rephrasing. if your rephrased question is not in the same language that the original question, translate it in the original question language.
    Make sure the rephrased standalone question is grammatically correct.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_CUSTOM_PROMPT = PromptTemplate.from_template(_template)

    prompt_template = """Tu es un agent spécialisé en assurance et ton rôle est d'aider les adhérents lorsqu'il se pose des questions.
    Si tu ne connais pas la réponse à la question, dit clairement que tu ne sais pas, et n'invente pas de réponse. Répond simplement et uniquement à la question posée.

    {context}

    Question: {question}

    Réponse: """
    QA_PROMPT_CUSTOM = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    MODEL_REPO_ID = "TheBloke/Llama-2-13B-chat-GGUF"
    MODEL_FILENAME = "llama-2-13b-chat.Q5_K_M.gguf"


    model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            resume_download=True,
            cache_dir=MODELS_PATH,
        )
    
    local_llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=32,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        temperature=0,
        max_tokens=2000,
        verbose=True,
        streaming=True
    )
 
    transformer_model = "all-MiniLM-L12-v2"
    model_embeddings = SentenceTransformerEmbeddings(model_name=transformer_model)
    vectorstore = Chroma(persist_directory=VECTORSTORE_CHROMADB_PATH, embedding_function=model_embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    compressor = LLMChainExtractor.from_llm(local_llm)
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)

    memory = ConversationBufferWindowMemory(
        input_key="question",
        output_key = "answer",
        memory_key="chat_history",
        return_messages= True,
        k=4
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm = local_llm,
        retriever=compression_retriever,
        rephrase_question = False,
        combine_docs_chain_kwargs=dict(prompt=QA_PROMPT_CUSTOM),
        condense_question_prompt= CONDENSE_QUESTION_CUSTOM_PROMPT,
        memory=memory,
        verbose= True,
        get_chat_history=lambda h : h,
        return_source_documents=True
        )

    return qa


@cl.on_chat_start
def main():
    chain = factory()
    cl.user_session.set("llm_chain", chain)

@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await llm_chain.acall({
        "question" : message.content,
        "chat_history" : []}, callbacks=[cb])

    print(res)
    await cl.Message(content=res["answer"]).send()
    return llm_chain
