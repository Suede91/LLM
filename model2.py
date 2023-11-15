from codetiming import Timer

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, pipeline
from transformers.pipelines.text_generation import TextGenerationPipeline

from huggingface_hub import hf_hub_download
from langchain.llms import HuggingFaceHub
from langchain.llms import LlamaCpp
from langchain.llms import HuggingFacePipeline
from auto_gptq import AutoGPTQForCausalLM

from langchain.vectorstores import Chroma, VectorStore
from langchain.embeddings import SentenceTransformerEmbeddings

from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferWindowMemory

from langchain.chains import LLMChain
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain

import importlib.util
import torch
import chainlit as cl

VECTORSTORE_CHROMADB_PATH = 'vectorstores/chromadb'
MODELS_PATH = 'models'

@Timer(name="init_model_hfhub")
def init_model_hfhub(repo_id: str) -> HuggingFaceHub:
    from getpass import getpass
    import os
    # HUGGINGFACEHUB_API_TOKEN = getpass()
    # os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
    local_llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.01, "max_length": 1024},
    )
    return local_llm

@Timer(name="init_model_gguf")
def init_model_gguf(repo_id: str, filename: str) -> LlamaCpp :
    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        resume_download=True,
        cache_dir=MODELS_PATH,
    )

    # callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    local_llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=32,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        temperature=0,
        max_tokens=2000,
        # callback_manager=callback_manager,
        verbose=True,
    )
    return local_llm

@Timer(name="init_model_gptq")
def init_model_gptq(repo_id: str) -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    pipe: TextGenerationPipeline = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        # max_length = MAX_NEW_TOKENS,
        temperature = 0.2,
        # top_p = 0.95,
        repetition_penalty = 1.15,
        # generation_config = generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@Timer(name="init_model_bin")
def init_model_bin(repo_id: str):
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
        repo_id
    )
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        repo_id,
        # torch_dtype=torch.float16,
        device_map='auto',
    )

    pipe: TextGenerationPipeline = pipeline(
        "text-generation",
        model = model,
        tokenizer = tokenizer,
        # max_length = MAX_NEW_TOKENS,
        temperature = 0.2,
        # top_p = 0.95,
        repetition_penalty = 1.15,
        # generation_config = generation_config,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)

    # # High-level wrapper
    # # https://api.python.langchain.com/en/latest/_modules/langchain/llms/huggingface_pipeline.html#HuggingFacePipeline.from_model_id
    # local_llm = HuggingFacePipeline.from_model_id(
    #     model_id = "bigscience/bloom-1b7",
    #     task = "text-generation",
    #     device = 0,
    #     model_kwargs = {"temperature": 0, "max_length": 64,  device_map='auto',},
    #     pipeline_kwargs = {""}
    # )

    return local_llm

def init_model(model_format: str, repo_id: str, filename: str):
    match model_format:
        case "HFHUB":
            return init_model_hfhub(
                repo_id = repo_id
            )
        case "GGUF":
            return init_model_gguf(
                repo_id = repo_id,
                filename = filename,
            )
        case "GPTQ":
            return init_model_gptq(
                repo_id = repo_id,
            )
        case "BIN":
            return init_model_bin(
                repo_id = repo_id,
            )
        case "GGML":
            raise ValueError(
                "GGML model format is not supported."
                "It has been deprecated and replaced by GGFU."
            )
        case "AWQ":
            raise ValueError(
                "AWQ model format is not yet supported."
            )
        case _:
            raise ValueError("Model type doesn't exist")
    
def build_llm():
    MODEL_FORMAT = "BIN"
    MODEL_REPO_ID = "bigscience/bloom-1b7"
    # MODEL_REPO_ID = "ehartford/dolphin-2.1-mistral-7b"
    MODEL_FILENAME = ""

    # MODEL_FORMAT = "GGUF"
    # MODEL_REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
    # MODEL_FILENAME = "mistral-7b-instruct-v0.1.Q5_K_M.gguf"

    MODEL_FORMAT = "GGUF"
    MODEL_REPO_ID = "TheBloke/Llama-2-13B-chat-GGUF"
    MODEL_FILENAME = "llama-2-13b-chat.Q5_K_M.gguf"


    # MODEL_FORMAT = "HFHUB"
    # MODEL_REPO_ID = "upstage/SOLAR-0-70b-16bit"
    # MODEL_REPO_ID = "Riiid/sheep-duck-llama-2-70b-v1.1"
    # MODEL_REPO_ID = "WizardLM/WizardLM-70B-V1.0"
    # MODEL_REPO_ID = "nomic-ai/gpt4all-falcon"
    # MODEL_REPO_ID = "google/flan-t5-xxl"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    # init_model(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    local_llm = init_model(
        model_format=MODEL_FORMAT, 
        repo_id=MODEL_REPO_ID, 
        filename=MODEL_FILENAME,
    )

def build_chain(local_llm):

    # template = """The following is a friendly conversation between a human and an AI.
    # The AI is talkative and provides lots of specific details from its context.
    # If the AI does not know the answer to a question, it truthfully says it does not know.

    # Current conversation:
    # {chat_history}
    # Human: {question}
    # Chatbot:"""
    template = """Tu es un agent spécialisé en assurance et ton rôle est d'aider les adhérents lorsqu'il se pose des questions.
    Si tu ne connais pas la réponse à la question, dit clairement que tu ne sais pas, et n'invente pas de réponse.

    {context}
    Historique de conversation:
    {chat_history}
    Question: {question}
    Reponse:"""

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=template,
    )

    template_condense = """
    A partir des fragments de conversation et de la question suivante, reformule le tout afin d'obtenir une question simplifiée, dans sa langue d'origine.

    Historique de conversation:
    {chat_history}
    Question: {question}
    Question simplifiée:"""

    prompt_condense = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template_condense,
    )

    memory = ConversationBufferWindowMemory(
        input_key="question",
        output_key = "answer",
        memory_key="chat_history",
        k=4,
    )

    transformer_model = "all-MiniLM-L6-v2"
    model_embeddings = SentenceTransformerEmbeddings(model_name=transformer_model)
    db = Chroma(persist_directory=VECTORSTORE_CHROMADB_PATH, embedding_function=model_embeddings)
    retriever = db.as_retriever(
        search_type="mmr", # "similarity", "similarity_score_threshold".
        search_kwargs={"k": 2},
        # k: Amount of documents to return (Default: 4)
        # score_threshold: Minimum relevance threshold for similarity_score_threshold
        # fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
        # lambda_mult: Diversity of results returned by MMR; 1 for minimum diversity and 0 for maximum. (Default: 0.5)
        # filter: Filter by document metadata
    )

    # docs = retriever.get_relevant_documents(query)

    # chain = LLMChain(
    #     llm=local_llm,
    #     prompt=prompt,
    #     memory=memory,
    #     verbose=True,
    # )

    # chain = ConversationChain(
    #     llm=local_llm,
    #     prompt=prompt,
    #     memory=memory,
    #     verbose=True,
    #     input_key = "question", # str = "input"  #: :meta private:
    #     # output_key: # str = "response"  #: :meta private:
    # )

    chain = ConversationalRetrievalChain.from_llm(
        # If different from default LLM and prompt
        # condense_question_llm=local_llm, # condense_question_llm: Optional[BaseLanguageModel] = None,
        condense_question_prompt=prompt_condense,

        memory=memory, # memory: Optional[BaseMemory] = None
        retriever=retriever, # retriever: BaseRetriever,
        chain_type = "stuff", # str = "stuff"
        llm=local_llm, # llm: BaseLanguageModel,
        combine_docs_chain_kwargs=dict(prompt=prompt), # Optional[Dict] = None,
        # callbacks: Callbacks = None,
        get_chat_history=lambda h : h,
        # verbose = True,
        return_generated_question = True,
        return_source_documents = True,
    )

#chainlit code
@cl.on_chat_start
async def start():
    # chain = qa_bot()
    local_llm = build_llm()
    chain = build_chain(local_llm)

    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to Medical Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()


