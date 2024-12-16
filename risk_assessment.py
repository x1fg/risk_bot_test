import faiss
import asyncio
import html
import requests
import logging

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
            "Содержит данные из полезных для анализа документов о клиенте: заключения служб, информация о клиенте, условиях кредитования."
        )
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

    async def generate_report(self, selected_risks, company_name, deal_details, message):
        results = {}

        for risk in selected_risks:
            agent_name = self.risk_to_agent_map.get(risk)
            if agent_name:
                try:
                    api_response = await self._call_api(risk, message)
                    api_info = self._extract_api_info(risk, api_response) if api_response else "Нет данных из API"

                    deal_prompt = f"""
                    Компания: {company_name}
                    Детали сделки: {deal_details}
                    Данные из API по {risk} риску: {api_info}
                    Анализ риска: {risk}
                    """
                    response = self.agents_dict[agent_name].invoke({"input": deal_prompt})
                    result_text = response.get("output", "Ошибка анализа.")
                    result_text = clean_formatting(result_text)
                    results[risk] = html.escape(result_text)

                except Exception as e:
                    results[risk] = html.escape(f"Ошибка: {str(e)}")

        report_lines = [f"💡 <b>Сделка:</b> {html.escape(clean_formatting(company_name))}\n"]
        report_lines.append(f"<b>Детали сделки:</b> {html.escape(clean_formatting(deal_details))}\n")

        for risk, result in results.items():
            report_lines.append(f"<b>{html.escape(clean_formatting(risk.capitalize()))} риск</b>")
            report_lines.append(f"Результат анализа:\n{result}\n")

        risk_table = self.generate_risk_table(results)
        report_text = "\n".join(report_lines)
        full_report = f"{report_text}\n<b>Сводная таблица рисков</b>:\n<pre>{html.escape(risk_table)}</pre>"

        report_chunks = split_message(full_report)
        for chunk in report_chunks:
            await message.answer(chunk, parse_mode="HTML")

    async def _call_api(self, risk, message):
        api_urls = {
            "финансовый": 'http://83.220.174.239:9797/rating',
            "репутационный": 'http://83.220.174.239:9797/news',
            "правовой": 'http://83.220.174.239:9797/juridical'
        }
        try:
            url = api_urls.get(risk)
            if url:
                await message.answer(f"🛠️ Вызов API для риска: {risk}") 
                response = requests.get(url, timeout=10)
                return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка при вызове API для риска '{risk}': {str(e)}")
            await message.answer(f"⚠️ Ошибка при вызове API для риска '{risk}': {str(e)}")
            return None
        
    def _extract_api_info(self, risk, api_response):
        if risk == "финансовый":
            value = api_response.get('value', 'Нет данных')
            return f"Оценка финансового риска: {value}/20"
        elif risk in ["репутационный", "правовой"]:
            return api_response.get('message', 'Нет новостей.')
        return "Нет данных от API"

    def generate_risk_table(self, risk_results):
        table_header = "Риск              | Уровень риска"
        table_separator = "-" * len(table_header)
        table_rows = []

        for risk, result in risk_results.items():
            risk_level = "Не определен"
            if "низкий" in result.lower():
                risk_level = "Низкий"
            elif "средний" in result.lower():
                risk_level = "Средний"
            elif "высокий" in result.lower():
                risk_level = "Высокий"

            table_rows.append(f"{risk.capitalize():<17} | {risk_level}")

        return "\n".join([table_header, table_separator, *table_rows])

def split_message(message, chunk_size=4000):
    if not isinstance(message, str):
        raise TypeError(f"split_message ожидает строку, а не объект типа {type(message)}")

    chunks = []
    while len(message) > chunk_size:
        split_index = message.rfind("\n", 0, chunk_size)
        if split_index == -1:
            split_index = chunk_size
        chunks.append(message[:split_index])
        message = message[split_index:].strip()
    chunks.append(message)
    return chunks

def clean_formatting(text):
    text = text.replace("**", "")
    text = text.replace("####", "")
    text = text.replace("###", "")
    return text.strip()