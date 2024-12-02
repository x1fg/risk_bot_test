import os
from dotenv import load_dotenv
from risk_assessment import DealAnalyzer
from bot import TelegramBot
from langchain_community.llms import GigaChat
from langchain_community.embeddings import GigaChatEmbeddings

load_dotenv()

if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")
    GIGACHAT_API_MODEL = os.getenv("GIGACHAT_API_MODEL", "GigaChat-Pro")
    GIGACHAT_API_SCOPE = os.getenv("GIGACHAT_API_SCOPE", "GIGACHAT_API_PERS")
    TOKENIZER_MODEL = "intfloat/multilingual-e5-large"

    DOCS_NAMING = {
        './data/docs/credit_inspection.txt': 'Заключение Кредитного Подразделения',
        './data/docs/reputation.txt': 'Заключение Подразделения Безопасности',
        './data/docs/prko.txt': 'ПРКО',
        './data/docs/law.txt': 'Заключение Юридического Подразделения',
        './data/docs/rm_conclusion.txt': 'Заключение Риск-Менеджера'
    }

    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("Отсутствует TELEGRAM_BOT_TOKEN в файле .env")

    llm = GigaChat(verify_ssl_certs=False, model=GIGACHAT_API_MODEL, scope=GIGACHAT_API_SCOPE, top_p=0.5)
    embeddings = GigaChatEmbeddings(verify_ssl_certs=False, scope=GIGACHAT_API_SCOPE)
    analyzer = DealAnalyzer(llm, embeddings, DOCS_NAMING, TOKENIZER_MODEL)

    bot = TelegramBot(TELEGRAM_BOT_TOKEN, analyzer)
    bot.run()