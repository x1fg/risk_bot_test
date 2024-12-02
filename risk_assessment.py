import faiss
import asyncio

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import BaseSingleActionAgent, AgentExecutor
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import BaseSingleActionAgent
from langchain.schema import AgentFinish
from pydantic import BaseModel
from transformers import AutoTokenizer

from prompts import prompts_dict


class CustomAgent(BaseSingleActionAgent, BaseModel):
    llm: object
    tools: list
    prompt_template: object

    @property
    def input_keys(self):
        return ["input"]

    @property
    def output_keys(self):
        return ["output"]

    def plan(self, intermediate_steps, **kwargs):
        input_data = kwargs["input"]
        prompt = self.prompt_template.format(input=input_data)
        response = self.llm(prompt)
        return AgentFinish({"output": response}, log="")

    async def aplan(self, intermediate_steps, **kwargs):
        return self.plan(intermediate_steps, **kwargs)


class DealAnalyzer:
    def __init__(self, llm, embeddings, docs_naming, tokenizer_model):
        self.llm = llm
        self.embeddings = embeddings
        self.docs_naming = docs_naming
        self.docs = []
        self.splitted_docs = []
        self.vector_store = None
        self.retriever_tool = None
        self.tools = []
        self.agents_dict = {}
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

        self.risk_to_agent_map = {
            "финансовый": "financial_expert",
            "маркетинговый": "marketing_expert",
            "репутационный": "reputation_expert",
            "правовой": "lawyer",
        }

        self.load_documents()
        self.prepare_vector_store()
        self.create_tools()
        self.create_agents()

    def load_documents(self):
        for txt_file_path in self.docs_naming:
            try:
                loader = TextLoader(txt_file_path, encoding="utf-8")
                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata.update({'source_type': self.docs_naming[txt_file_path]})
                self.docs.extend(file_docs)
            except FileNotFoundError:
                print(f"File not found: {txt_file_path}")
            except UnicodeDecodeError:
                print(f"Encoding issue with file: {txt_file_path}")
            except Exception as e:
                print(f"Error loading file {txt_file_path}: {e}")

    def prepare_vector_store(self):
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer, chunk_size=512, chunk_overlap=128)
        self.splitted_docs = text_splitter.split_documents(self.docs)
        index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        self.vector_store.add_documents(documents=self.splitted_docs)

    def create_tools(self):
        retriever = self.vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        self.retriever_tool = create_retriever_tool(
            retriever,
            "deal_docs_search",
            "Содержит данные из полезных для анализа документов о клиенте: заключения служб, информация о клиенте, условиях кредитования.")
        search_tool = TavilySearchResults()
        self.tools = [self.retriever_tool, search_tool]

    def create_agents(self):
        agents_names = ['marketing_expert', 'lawyer', 'reputation_expert', 'financial_expert']
        for agent_name in agents_names:
            agent_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", prompts_dict[f'system_{agent_name}']),
                    ("human", "Вопрос: {input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ]
            )
            agent = CustomAgent(llm=self.llm, tools=self.tools, prompt_template=agent_prompt)
            agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=False)
            self.agents_dict[agent_name] = agent_executor

    async def generate_report(self, selected_risks, company_name, deal_details):
        results = {}
        for risk in selected_risks:
            agent_name = self.risk_to_agent_map.get(risk)
            if agent_name:
                try:
                    deal_prompt = f"Компания: {company_name}\nДетали сделки: {deal_details}\nАнализ риска: {risk}"
                    agent_executor = self.agents_dict[agent_name]
                    if callable(getattr(agent_executor, "invoke", None)):
                        if asyncio.iscoroutinefunction(agent_executor.invoke):
                            response = await agent_executor.invoke({"input": deal_prompt})
                        else:
                            response = agent_executor.invoke({"input": deal_prompt})
                    else:
                        raise ValueError(f"Ошибка invoke {agent_name}")
                    
                    results[risk] = response.get("output", "Ошибка анализа.")
                except Exception as e:
                    results[risk] = f"Ошибка: {str(e)}"
        report = "\n\n".join(
            [f"Риск: {risk}\nРезультат:\n{result}" for risk, result in results.items()]
        )
        return report