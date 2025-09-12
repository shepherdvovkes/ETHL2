# 🚀 Polygon Price Prediction & Analysis System

Комплексная система для анализа сети Polygon и предсказания цены MATIC на следующую неделю.

## 📊 Что анализируется

### 1. **On-Chain метрики сети Polygon**
- **TVL (Total Value Locked)** - общая заблокированная стоимость в DeFi протоколах
- **Активность сети**: количество транзакций, активные адреса, новые адреса
- **Gas метрики**: средняя цена газа, использование сети, эффективность
- **DeFi активность**: объемы в протоколах (Aave, QuickSwap, Curve, SushiSwap)
- **Bridge активность**: объемы кроссчейн переводов

### 2. **Финансовые метрики MATIC**
- **Ценовые данные**: исторические цены, волатильность, объемы торгов
- **Рыночные метрики**: рыночная капитализация, доминирование, ликвидность
- **Технические индикаторы**: RSI, MACD, Moving Averages
- **Корреляции**: с ETH, BTC, другими Layer 2 токенами

### 3. **Экосистемные метрики**
- **Разработка**: активность GitHub, количество коммитов, новые проекты
- **Партнерства**: новые интеграции, листинги на биржах
- **Сообщество**: активность в социальных сетях, количество пользователей
- **Adoption**: рост числа dApps, пользователей, транзакций

### 4. **Макроэкономические факторы**
- **Общий рынок**: индекс страха и жадности, настроения рынка
- **Регуляторные новости**: влияние регуляций на крипто
- **Технологические обновления**: обновления Ethereum, конкуренты Layer 2

### 5. **Специфичные для Polygon факторы**
- **Polygon zkEVM**: активность и adoption
- **Polygon ID**: использование identity решений
- **Polygon Supernets**: развитие приватных сетей
- **Staking метрики**: количество валидаторов, стейкинг APY

## 🛠️ Компоненты системы

### 1. **Сбор данных** (`polygon_price_prediction_data.py`)
- Сбор on-chain метрик через QuickNode API
- Получение финансовых данных через CoinGecko API
- Анализ DeFi протоколов на Polygon
- Сбор метрик сообщества и разработки

### 2. **ML предсказание** (`polygon_price_predictor.py`)
- 8 различных ML моделей (Random Forest, XGBoost, Neural Networks, etc.)
- Ensemble предсказания с весовыми коэффициентами
- Feature engineering и технические индикаторы
- Валидация и кросс-валидация моделей

### 3. **Анализ трендов** (`polygon_trend_analyzer.py`)
- Анализ ценовых трендов
- Обнаружение сезонных паттернов
- Выявление аномалий
- Корреляционный анализ

### 4. **Главный пайплайн** (`run_polygon_analysis.py`)
- Объединяет все компоненты
- Генерирует комплексный отчет
- Предоставляет рекомендации

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Настройка конфигурации
Убедитесь, что в `config.env` настроены:
```env
QUICKNODE_API_KEY=your_quicknode_api_key
QUICKNODE_HTTP_ENDPOINT=your_quicknode_endpoint
COINGECKO_API_KEY=your_coingecko_api_key
```

### 3. Инициализация базы данных
```bash
python setup_database_v2.py
```

### 4. Запуск полного анализа
```bash
python run_polygon_analysis.py
```

## 📋 Пошаговое использование

### Шаг 1: Сбор данных
```bash
python polygon_price_prediction_data.py
```
Собирает все необходимые данные о Polygon и сохраняет в базу данных.

### Шаг 2: Анализ трендов
```bash
python polygon_trend_analyzer.py
```
Анализирует тренды и паттерны, сохраняет результаты в `polygon_trend_analysis.json`.

### Шаг 3: Предсказание цены
```bash
python polygon_price_predictor.py
```
Обучает ML модели и делает предсказание цены на 7 дней.

### Шаг 4: Полный анализ
```bash
python run_polygon_analysis.py
```
Запускает все компоненты и генерирует итоговый отчет.

## 📊 Результаты анализа

### 1. **Трендовый анализ**
- Общий тренд (bullish/bearish/sideways)
- Распределение трендов по метрикам
- Обнаруженные паттерны и аномалии
- Корреляции между метриками

### 2. **ML предсказание**
- Предсказание изменения цены на 7 дней
- Уровень уверенности модели
- Важность различных факторов
- Производительность моделей

### 3. **Рекомендации**
- Торговые рекомендации
- Управление рисками
- Мониторинг ключевых метрик
- Стратегии входа/выхода

## 📈 Пример вывода

```
🎯 POLYGON ANALYSIS SUMMARY REPORT
================================================================================
📊 Analysis Status: COMPLETED
📅 Analysis Period: 90 days
🔮 Prediction Horizon: 7 days
📈 Overall Trend: BULLISH
   • Bullish: 65.2%
   • Bearish: 20.1%
   • Sideways: 14.7%
💰 Current MATIC Price: $0.8234
🔮 Predicted 7-day Change: +8.5% (Confidence: 78%)
🎯 Predicted Price: $0.8934

💡 KEY RECOMMENDATIONS:
   1. Overall bullish trend detected. Consider holding or accumulating MATIC.
   2. High confidence bullish prediction. Consider accumulating MATIC.
   3. Growing TVL indicates increasing DeFi adoption. Positive for long-term price.
   4. Monitor on-chain metrics for early signals of trend changes.
   5. Watch for DeFi TVL growth as a positive indicator for MATIC price.
```

## 🔧 Настройка и кастомизация

### Настройка периодов анализа
В скриптах можно изменить параметры:
```python
# Период сбора данных
days_back = 90  # дней назад

# Период обучения ML моделей
training_days = 180  # дней для обучения

# Горизонт предсказания
prediction_horizon = "7d"  # 1d, 7d, 30d
```

### Добавление новых метрик
1. Добавить новые поля в модели базы данных
2. Обновить сборщик данных
3. Включить в ML pipeline
4. Добавить в анализ трендов

### Настройка ML моделей
В `polygon_price_predictor.py` можно:
- Добавить новые модели
- Изменить гиперпараметры
- Настроить веса ensemble
- Добавить новые признаки

## 📁 Структура файлов

```
QUICKNODEANALYTICS/
├── polygon_price_prediction_data.py    # Сбор данных
├── polygon_price_predictor.py          # ML предсказание
├── polygon_trend_analyzer.py           # Анализ трендов
├── run_polygon_analysis.py             # Главный пайплайн
├── polygon_analysis_report.json        # Итоговый отчет
├── polygon_trend_analysis.json         # Анализ трендов
└── models/polygon/                     # Сохраненные ML модели
    ├── random_forest_model.joblib
    ├── xgboost_model.joblib
    ├── neural_network_model.joblib
    └── feature_importance.json
```

## ⚠️ Важные замечания

### 1. **Ограничения предсказаний**
- Криптовалютные рынки крайне волатильны
- Предсказания основаны на исторических данных
- Внешние события могут кардинально изменить тренды
- Используйте предсказания как дополнительный инструмент

### 2. **Управление рисками**
- Никогда не инвестируйте больше, чем можете позволить себе потерять
- Диверсифицируйте портфель
- Используйте stop-loss ордера
- Проводите собственное исследование

### 3. **Технические требования**
- Стабильное интернет-соединение
- Достаточно места на диске для данных
- API ключи для внешних сервисов
- Регулярное обновление данных

## 🔄 Автоматизация

### Настройка cron для регулярного анализа
```bash
# Ежедневный анализ в 9:00
0 9 * * * cd /path/to/QUICKNODEANALYTICS && python run_polygon_analysis.py

# Еженедельный полный анализ в воскресенье в 10:00
0 10 * * 0 cd /path/to/QUICKNODEANALYTICS && python run_polygon_analysis.py
```

### Мониторинг и алерты
Система может быть интегрирована с:
- Telegram боты для уведомлений
- Email алерты
- Slack интеграции
- Webhook уведомления

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи в консоли
2. Убедитесь в правильности API ключей
3. Проверьте подключение к базе данных
4. Обратитесь к документации API сервисов

## 📚 Дополнительные ресурсы

- [Polygon Documentation](https://docs.polygon.technology/)
- [QuickNode API Docs](https://www.quicknode.com/docs/)
- [CoinGecko API Docs](https://www.coingecko.com/en/api)
- [Machine Learning Best Practices](https://scikit-learn.org/stable/)

---

**Disclaimer**: Данная система предназначена для образовательных и исследовательских целей. Не является финансовым советом. Всегда проводите собственное исследование перед принятием инвестиционных решений.
