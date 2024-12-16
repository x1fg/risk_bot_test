from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters.command import Command
from aiogram import Router
import requests
import logging


class DealForm(StatesGroup):
    company_name = State()
    deal_details = State()


class TelegramBot:
    def __init__(self, token, analyzer):
        self.bot = Bot(token=token)
        self.storage = MemoryStorage()
        self.dp = Dispatcher(storage=self.storage)
        self.router = Router()
        self.analyzer = analyzer
        self._register_handlers()

    def _register_handlers(self):
        @self.router.message(Command("start"))
        async def start(message: types.Message, state: FSMContext):
            await state.set_state(DealForm.company_name)
            await message.answer("Введите название компании:")

        @self.router.message(DealForm.company_name)
        async def get_company_name(message: types.Message, state: FSMContext):
            company_name = message.text.strip()
            await state.update_data(company_name=company_name)
            await state.set_state(DealForm.deal_details)
            await message.answer("Введите детали сделки:")

        @self.router.message(DealForm.deal_details)
        async def get_deal_details(message: types.Message, state: FSMContext):
            deal_details = message.text.strip()
            data = await state.get_data()
            company_name = data.get("company_name", "Неизвестная компания")
            selected_risks = list(self.analyzer.risk_to_agent_map.keys())
            await self.analyzer.generate_report(selected_risks, company_name, deal_details, message)

    def run(self):
        self.dp.include_router(self.router)
        self.dp.run_polling(self.bot, skip_updates=True)