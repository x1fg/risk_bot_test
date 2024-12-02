from aiogram import Bot, Dispatcher, types, F
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters.command import Command
from aiogram import Router


class DealForm(StatesGroup):
    company_name = State()
    deal_details = State()
    risk_selection = State()


class TelegramBot:
    def __init__(self, token, analyzer):
        self.bot = Bot(token=token)
        self.storage = MemoryStorage()
        self.dp = Dispatcher(storage=self.storage)
        self.router = Router()
        self.analyzer = analyzer
        self.risk_to_agent_map = {
            "финансовый": "financial_expert",
            "маркетинговый": "marketing_expert",
            "репутационный": "reputation_expert",
            "правовой": "lawyer",
        }
        self._register_handlers()

    def _register_handlers(self):
        @self.router.message(Command("start"))
        async def start(message: types.Message, state: FSMContext):
            await state.set_state(DealForm.company_name)
            await message.answer("Введите название компании, риски которой хотите проанализировать:")

        @self.router.message(DealForm.company_name)
        async def get_company_name(message: types.Message, state: FSMContext):
            company_name = message.text.strip()
            await state.update_data(company_name=company_name)
            await state.set_state(DealForm.deal_details)
            await message.answer("Введите подробности сделки (сумма, цель, срок и т.д.):")

        @self.router.message(DealForm.deal_details)
        async def get_deal_details(message: types.Message, state: FSMContext):
            deal_details = message.text.strip()
            await state.update_data(deal_details=deal_details)
            await state.set_state(DealForm.risk_selection)
            keyboard = self._get_risk_keyboard()
            await message.answer(
                "Выберите риски для анализа (можно выбрать несколько):",
                reply_markup=keyboard
            )

        @self.router.callback_query(F.data.startswith("risk_"))
        async def risk_selection(callback_query: types.CallbackQuery, state: FSMContext):
            selected_risk = callback_query.data.split("_")[1]
            data = await state.get_data()
            selected_risks = data.get("selected_risks", [])
            if selected_risk in selected_risks:
                selected_risks.remove(selected_risk)
            else:
                selected_risks.append(selected_risk)
            await state.update_data(selected_risks=selected_risks)
            keyboard = self._get_risk_keyboard(selected_risks=selected_risks)
            await callback_query.message.edit_reply_markup(reply_markup=keyboard)
            await callback_query.answer()

        @self.router.callback_query(F.data == "done")
        async def generate_report(callback_query: types.CallbackQuery, state: FSMContext):
            data = await state.get_data()
            company_name = data.get("company_name", "Неизвестная компания")
            deal_details = data.get("deal_details", "Нет данных о сделке")
            selected_risks = data.get("selected_risks", [])
            if not selected_risks:
                await callback_query.answer("Вы не выбрали ни одного риска!", show_alert=True)
                return
            await callback_query.answer("Анализирую риски, пожалуйста, подождите...")
            report = await self.analyzer.generate_report(selected_risks, company_name, deal_details)
            await callback_query.message.answer(f"Итоговый отчет:\n\n{report}")
            await state.clear()

    def _get_risk_keyboard(self, selected_risks=None):
        if selected_risks is None:
            selected_risks = []
        risks = {
            "финансовый": "Финансовый риск",
            "маркетинговый": "Маркетинговый риск",
            "репутационный": "Риск деловой репутации",
            "правовой": "Правовой риск",
        }
        buttons = [
            InlineKeyboardButton(
                text=f"{'✔️ ' if key in selected_risks else '✖️ '}{label}",
                callback_data=f"risk_{key}"
            )
            for key, label in risks.items()
        ]
        buttons.append(InlineKeyboardButton(text="Готово", callback_data="done"))
        return InlineKeyboardMarkup(inline_keyboard=[[button] for button in buttons])

    def run(self):
        self.dp.include_router(self.router)
        self.dp.run_polling(self.bot, skip_updates=True)