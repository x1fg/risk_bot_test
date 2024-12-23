from dotenv import load_dotenv
from risk_assessment import DealAnalyzer
from bot import TelegramBot
from langchain_community.embeddings import GigaChatEmbeddings
from api_caller import APICaller
import os

load_dotenv()

if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = os.getenv("BOT_TOKEN")
    GIGACHAT_API_SCOPE = os.getenv("GIGACHAT_API_SCOPE", "GIGACHAT_API_PERS")
    GIGACHAT_API_MODEL = os.getenv("GIGACHAT_API_MODEL", "GigaChat-Pro")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    TOKENIZER_MODEL = "intfloat/multilingual-e5-large"

    DOCS_NAMING = {
        './data/docs/credit_inspection.txt': 'Заключение Кредитного Подразделения',
        './data/docs/reputation.txt': 'Заключение Подразделения Безопасности',
        './data/docs/prko.txt': 'ПРКО',
        './data/docs/law.txt': 'Заключение Юридического Подразделения',
        './data/docs/rm_conclusion.txt': 'Заключение Риск-Менеджера'
    }

    if not TELEGRAM_BOT_TOKEN or not OPENAI_API_KEY:
        raise ValueError("Отсутствует BOT_TOKEN или OPENAI_API_KEY в файле .env")

    embeddings = GigaChatEmbeddings(verify_ssl_certs=False, scope=GIGACHAT_API_SCOPE)

    llm = APICaller(api_key=OPENAI_API_KEY)
    analyzer = DealAnalyzer(llm, embeddings, DOCS_NAMING, TOKENIZER_MODEL)
    bot = TelegramBot(TELEGRAM_BOT_TOKEN, analyzer)
    bot.run()