# 🚀 DEFIMON Analytics Backend v2.0

Продвинутый Backend для анализа криптовалютных данных с машинным обучением.

## ✨ Основные возможности

### 🔌 Подключение к внешним API
- **QuickNode** - для данных блокчейна (Ethereum, Polygon, Arbitrum, Optimism)
- **Etherscan** - для анализа смарт-контрактов и транзакций
- **CoinGecko** - для финансовых данных и цен
- **GitHub** - для метрик разработки (с веб-авторизацией)
- **Универсальный Blockchain Client** - поддержка 50+ блокчейнов

### 📊 Система метрик
- **10 категорий метрик**: On-chain, Financial, GitHub, Tokenomics, Security, Community, Partnership, Network, Trending, Cross-chain
- **50+ метрик** с автоматическим маппингом источников данных
- **Умная система определения** какие данные загружать для каждой метрики

### 📈 Загрузка данных
- **Гибкие временные периоды**: 1 неделя, 2 недели, 4 недели
- **Автоматическая загрузка** для выбранной сети или актива
- **Фоновые задачи** для непрерывного сбора данных
- **Обработка ошибок** и повторные попытки

### 🤖 Машинное обучение
- **6 алгоритмов ML**: Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, Neural Network
- **Автоматическая подготовка данных** и feature engineering
- **Ensemble предсказания** для повышения точности
- **Инвестиционные оценки** с confidence scores

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEFIMON Backend v2.0                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   FastAPI       │    │   Data Loader   │    │ ML Pipeline  │ │
│  │   (REST API)    │◄──►│   (Background)  │◄──►│ (Predictions)│ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                       │     │
│           │                       │                       │     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   PostgreSQL    │    │   Metrics       │    │ Auth Handler │ │
│  │   (Database)    │◄──►│   Mapper        │◄──►│ (OAuth)      │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    External APIs                                │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   QuickNode     │    │   CoinGecko     │    │   GitHub     │ │
│  │   (Blockchain)  │    │   (Financial)   │    │ (Development)│ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка окружения

Скопируйте `config.env` и настройте переменные:

```bash
cp config.env .env
```

Обязательные переменные:
```env
QUICKNODE_API_KEY=your_quicknode_key
QUICKNODE_HTTP_ENDPOINT=your_quicknode_endpoint
DATABASE_URL=postgresql://user:password@localhost:5432/defimon_db
```

Опциональные переменные:
```env
ETHERSCAN_API_KEY=your_etherscan_key
COINGECKO_API_KEY=your_coingecko_key
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
HF_TOKEN=your_huggingface_token
```

### 3. Инициализация базы данных

```bash
python setup_database_v2.py
```

### 4. Запуск Backend

```bash
python run_backend.py
```

Backend будет доступен по адресу: http://localhost:8000

## 📚 API Документация

### Основные эндпоинты

- **GET /** - Главная страница с информацией
- **GET /docs** - Swagger UI документация
- **GET /redoc** - ReDoc документация
- **GET /api/health** - Проверка состояния

### Управление активами

```bash
# Получить список активов
GET /api/assets

# Создать новый актив
POST /api/assets
{
  "symbol": "MATIC",
  "name": "Polygon",
  "contract_address": "0x0000000000000000000000000000000000001010",
  "blockchain_id": 1,
  "coingecko_id": "matic-network",
  "github_repo": "maticnetwork/polygon-sdk"
}
```

### Сбор данных

```bash
# Собрать данные для актива
POST /api/collect-data
{
  "asset_id": 1,
  "time_periods": ["1w", "2w", "4w"],
  "metrics": ["price_usd", "tvl", "commits_7d"]
}

# Собрать данные для всей сети
POST /api/collect-data
{
  "blockchain_id": 1,
  "time_periods": ["1w", "2w", "4w"]
}
```

### Машинное обучение

```bash
# Обучить модели
POST /api/train-models

# Получить предсказание
POST /api/predict
{
  "asset_id": 1,
  "model_name": "ensemble"
}

# Получить все предсказания
GET /api/predictions
```

### Авторизация GitHub

```bash
# Начать авторизацию
GET /auth/github

# Проверить статус авторизации
GET /auth/github/status
```

## 🔧 Конфигурация

### Поддерживаемые блокчейны

Backend поддерживает 50+ блокчейнов:

**Layer 1**: Ethereum, Bitcoin, BSC, Polygon, Avalanche, Solana, Cardano, Polkadot, Cosmos, Fantom

**Layer 2**: Arbitrum, Optimism, Polygon zkEVM, zkSync Era, StarkNet, Base, Linea, Scroll, Mantle, Blast

**DeFi**: Uniswap, Aave, Compound, Curve, SushiSwap

**Gaming & NFT**: Immutable X, Axie Infinity, The Sandbox, Decentraland

### Категории метрик

1. **On-chain** - TVL, транзакции, активные адреса
2. **Financial** - цена, объем, волатильность
3. **GitHub** - коммиты, звезды, форки, issues
4. **Tokenomics** - предложение, инфляция, burning
5. **Security** - аудит, верификация контрактов
6. **Community** - социальные сети, engagement
7. **Partnership** - интеграции, листинги
8. **Network** - время блоков, утилизация сети
9. **Trending** - momentum, sentiment, fear/greed
10. **Cross-chain** - мосты, кроссчейн ликвидность

## 🤖 Машинное обучение

### Алгоритмы

- **Random Forest** - для robust предсказаний
- **Gradient Boosting** - для сложных паттернов
- **Linear Regression** - baseline модель
- **Ridge/Lasso** - регуляризованные модели
- **Neural Network** - deep learning
- **Ensemble** - комбинация всех моделей

### Feature Engineering

- **Временные признаки** - лаги, скользящие средние
- **Интерактивные признаки** - комбинации метрик
- **Нормализация** - StandardScaler, MinMaxScaler
- **Обработка пропусков** - медиана, интерполяция

### Предсказания

- **Investment Score** - общая оценка инвестиционной привлекательности
- **Price Prediction** - предсказание цены
- **Risk Assessment** - оценка рисков
- **Confidence Score** - уверенность в предсказании

## 📊 Мониторинг

### Логи

Логи сохраняются в:
- `logs/backend.log` - основные логи
- Консоль - INFO уровень и выше

### Метрики

- Количество собранных данных
- Производительность ML моделей
- Статус внешних API
- Ошибки и исключения

## 🔒 Безопасность

### API Keys

- Все ключи хранятся в переменных окружения
- Никогда не коммитятся в репозиторий
- Используется `.env` файл для локальной разработки

### GitHub OAuth

- Веб-авторизация через OAuth 2.0
- Безопасное хранение токенов
- Автоматическое обновление токенов

### Rate Limiting

- Встроенные ограничения для всех API
- Автоматические задержки между запросами
- Обработка ошибок rate limiting

## 🚀 Развертывание

### Docker

```bash
# Сборка образа
docker build -t defimon-backend .

# Запуск контейнера
docker run -p 8000:8000 --env-file .env defimon-backend
```

### Production

```bash
# Запуск с Gunicorn
gunicorn src.api.backend_api:app -w 4 -k uvicorn.workers.UvicornWorker

# Запуск с PM2
pm2 start run_backend.py --name defimon-backend
```

## 🐛 Отладка

### Проверка подключений

```bash
# Проверка базы данных
python -c "from src.database.database import init_db; init_db()"

# Проверка API ключей
python -c "from src.config.settings import settings; print(settings.QUICKNODE_API_KEY)"

# Тест внешних API
curl http://localhost:8000/api/health
```

### Логи ошибок

```bash
# Просмотр логов
tail -f logs/backend.log

# Фильтр ошибок
grep "ERROR" logs/backend.log
```

## 📈 Производительность

### Оптимизация

- **Асинхронные запросы** - параллельная обработка
- **Кэширование** - Redis для часто используемых данных
- **Batch обработка** - группировка запросов
- **Connection pooling** - переиспользование соединений

### Масштабирование

- **Горизонтальное** - несколько инстансов API
- **Вертикальное** - увеличение ресурсов
- **База данных** - read replicas для аналитики
- **Кэш** - Redis cluster для высокой доступности

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📄 Лицензия

MIT License - см. файл LICENSE

## 🆘 Поддержка

- **Issues** - GitHub Issues для багов и предложений
- **Discussions** - GitHub Discussions для вопросов
- **Documentation** - подробная документация в `/docs`

---

**DEFIMON Analytics Backend v2.0** - Продвинутая аналитика криптовалют с машинным обучением 🚀
