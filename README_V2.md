# 🚀 DEFIMON v2 - Advanced Crypto Analytics System

## 🌟 Обзор

DEFIMON v2 - это комплексная аналитическая система для оценки инвестиционной привлекательности крипто-активов, поддерживающая **50+ блокчейнов** и **10 категорий метрик**.

## ✨ Ключевые возможности

### 🌐 Мульти-блокчейн поддержка
- **50+ блокчейнов**: Ethereum, Polygon, BSC, Arbitrum, Solana, Cardano, Polkadot, Cosmos, и многие другие
- **Layer 1 & Layer 2**: Основные сети и решения масштабирования
- **DeFi протоколы**: Uniswap, Aave, Compound, Curve, SushiSwap
- **Gaming & NFT**: Immutable X, Axie Infinity, The Sandbox, Decentraland
- **Privacy coins**: Monero, Zcash, Dash
- **Enterprise блокчейны**: Hyperledger, Corda

### 📊 Комплексные метрики (10 категорий)

#### 1. 🔗 On-Chain метрики
- **TVL**: Total Value Locked и изменения
- **Транзакции**: Объем, количество, комиссии
- **Пользователи**: Активные адреса, новые пользователи, удержание
- **Смарт-контракты**: Развертывание, взаимодействия, сложность
- **DeFi метрики**: Пулы ликвидности, yield farming, кредитование

#### 2. 💰 Финансовые метрики
- **Капитализация**: Рыночная, полностью разбавленная
- **Цены**: Текущая, исторические максимумы/минимумы
- **Объемы**: 24h, 7d, изменения
- **Волатильность**: 24h, 7d, 30d, бета-коэффициент
- **Ликвидность**: Спреды, глубина ордербука, проскальзывание

#### 3. 🪙 Токеномика
- **Предложение**: Циркулирующее, общее, максимальное
- **Распределение**: Команда, инвесторы, сообщество, казначейство
- **Vesting**: График разблокировки, следующие разблокировки
- **Утилитарность**: Оценка полезности, управление, стейкинг

#### 4. 👨‍💻 GitHub активность
- **Разработка**: Коммиты, PR, issues, качество кода
- **Сообщество**: Контрибьюторы, звезды, форки
- **Языки**: Основной язык, распределение языков
- **Метрики**: Покрытие тестами, время решения issues

#### 5. 🔒 Безопасность
- **Аудит**: Статус, компания, дата, оценка
- **Контракты**: Верификация, исходный код, уязвимости
- **Децентрализация**: Управление, валидаторы, узлы
- **Защита**: Reentrancy, overflow, access control

#### 6. 👥 Сообщество
- **Социальные сети**: Twitter, Telegram, Discord, Reddit
- **Engagement**: Уровень вовлеченности, активность
- **Контент**: Блоги, видео, подкасты, упоминания в СМИ
- **Образование**: Документация, туториалы, поддержка

#### 7. 🤝 Партнерства
- **Стратегические**: Количество, качество, уровень
- **Интеграции**: Биржи, кошельки, DeFi протоколы
- **Экосистема**: Связи, совместимость, принятие

#### 8. 🌐 Сетевые метрики
- **Производительность**: Время блока, TPS, утилизация
- **Безопасность**: Хешрейт, сложность, валидаторы
- **Экономика**: Предложение, инфляция, сжигание
- **Газ**: Цены, лимиты, использование

#### 9. 📈 Трендовые метрики
- **Импульс**: Оценка импульса, направление тренда
- **Сезонность**: Сезонные паттерны, циклы
- **Аномалии**: Обнаружение аномалий, типы, серьезность
- **Настроения**: Fear & Greed, социальные, новостные

#### 10. 🔄 Кроссчейн метрики
- **Мосты**: Объемы, транзакции, комиссии
- **Ликвидность**: Кроссчейн ликвидность, дисбаланс

### 🤖 ML и предсказания
- **Инвестиционный балл**: Комплексная оценка (0-1)
- **Прогнозы цен**: 1d, 7d, 30d с доверительными интервалами
- **Оценка риска**: Анализ рисков и волатильности
- **Потенциал роста**: Оценка потенциала развития
- **Модели**: Random Forest, Gradient Boosting, Neural Networks, Linear Regression

## 🏗️ Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEFIMON v2 Analytics                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Web Interface │    │   API Server    │    │ Data Collector│ │
│  │   (React/HTML)  │◄──►│   (FastAPI)     │◄──►│   (Worker)   │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│           │                       │                       │     │
│           │                       │                       │     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   PostgreSQL    │    │     Redis       │    │ ML Pipeline  │ │
│  │   (Database)    │◄──►│   (Cache)       │◄──►│ (PyTorch)    │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    External APIs (50+ sources)                 │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   QuickNode     │    │   Etherscan     │    │Hugging Face  │ │
│  │   (Multi-chain) │    │   (Multi-chain) │    │   (Models)   │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   CoinGecko     │    │   GitHub API    │    │Social APIs   │ │
│  │   (Market Data) │    │   (Development) │    │(Community)   │ │
│  └─────────────────┘    └─────────────────┘    └──────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Быстрый старт

### 1. Установка зависимостей
```bash
# Клонирование репозитория
git clone <repository>
cd QUICKNODEANALYTICS

# Установка Python зависимостей
pip install -r requirements.txt

# Установка Node.js зависимостей (если нужно)
npm install
```

### 2. Настройка конфигурации
```bash
# Копирование конфигурации
cp config.env.example config.env

# Редактирование конфигурации
nano config.env
```

### 3. Запуск системы
```bash
# Автоматическая настройка и запуск
python run_defimon_v2.py
```

### 4. Доступ к системе
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📊 API Endpoints

### Assets
- `GET /api/assets` - Список всех активов
- `GET /api/assets/{id}` - Детали актива с метриками
- `GET /api/assets/{id}/metrics` - Все метрики актива

### Analytics
- `GET /api/analytics` - Обзор системы
- `GET /api/analytics/{asset_id}` - Аналитика по активу
- `GET /api/analytics/compare` - Сравнение активов

### Predictions
- `GET /api/predictions` - Все предсказания
- `GET /api/predictions/{asset_id}` - Предсказания по активу
- `POST /api/predict` - Создать новое предсказание

### Blockchains
- `GET /api/blockchains` - Поддерживаемые блокчейны
- `GET /api/blockchains/{id}/metrics` - Метрики блокчейна

### Competitors
- `GET /api/competitors` - Анализ конкурентов
- `GET /api/competitors/compare` - Сравнение с конкурентами

### Management
- `POST /api/retrain` - Переобучение ML моделей
- `GET /api/stats` - Статистика сбора данных
- `GET /api/health` - Проверка здоровья системы

## 🗄️ База данных

### PostgreSQL (Рекомендуется)
```bash
# Создание базы данных
createdb defimon_db

# Настройка пользователя
psql defimon_db
CREATE USER defimon WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE defimon_db TO defimon;
```

### SQLite (Разработка)
```bash
# Автоматическое создание
python setup_database_v2.py
```

### Миграция
```bash
# Миграция с существующей БД
python migrate_database.py
```

## 🔧 Конфигурация

### Переменные окружения
```bash
# API Keys
QUICKNODE_API_KEY=your_quicknode_key
ETHERSCAN_API_KEY=your_etherscan_key
GITHUB_TOKEN=your_github_token
COINGECKO_API_KEY=your_coingecko_key
HF_TOKEN=your_huggingface_token

# Database
DATABASE_URL=postgresql://defimon:password@localhost:5432/defimon_db
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Data Collection
COLLECTION_INTERVAL=3600  # seconds
BATCH_SIZE=100

# ML Configuration
ML_MODEL_PATH=./models
HUGGINGFACE_MODEL=microsoft/DialoGPT-medium
```

## 📈 Метрики в деталях

### On-Chain метрики
```python
{
    "tvl": 1000000.0,                    # Total Value Locked
    "tvl_change_24h": 5.2,               # TVL change 24h %
    "daily_transactions": 15000,         # Daily transactions
    "active_addresses_24h": 5000,        # Active addresses
    "gas_price_avg": 0.001,              # Average gas price
    "network_utilization": 75.5          # Network utilization %
}
```

### Финансовые метрики
```python
{
    "price_usd": 1.25,                   # Price in USD
    "market_cap": 50000000.0,            # Market capitalization
    "volume_24h": 2000000.0,             # Volume 24h
    "volatility_24h": 12.5,              # Volatility 24h %
    "price_change_24h": 3.2              # Price change 24h %
}
```

### GitHub метрики
```python
{
    "commits_30d": 150,                  # Commits last 30 days
    "stars": 2500,                       # GitHub stars
    "active_contributors_30d": 12,       # Active contributors
    "code_quality_score": 8.5,           # Code quality score
    "test_coverage": 85.0                # Test coverage %
}
```

## 🤖 ML Модели

### Инвестиционный балл
```python
{
    "investment_score": 0.75,            # Overall score (0-1)
    "confidence": 0.85,                  # Confidence level
    "factors": {
        "tvl_growth": 0.25,              # TVL growth weight
        "price_performance": 0.20,       # Price performance weight
        "development_activity": 0.20,    # Development activity weight
        "volume_activity": 0.15,         # Volume activity weight
        "user_activity": 0.10,           # User activity weight
        "community_engagement": 0.10     # Community engagement weight
    }
}
```

### Прогнозы цен
```python
{
    "price_prediction_7d": 1.35,         # 7-day price prediction
    "price_prediction_30d": 1.50,        # 30-day price prediction
    "confidence_interval": [1.20, 1.60], # Confidence interval
    "model_version": "v2.1.0"            # Model version
}
```

## 🔍 Мониторинг

### Логи
```bash
# Application logs
tail -f logs/defimon.log

# Error logs
tail -f logs/error.log

# Performance logs
tail -f logs/performance.log
```

### Метрики системы
```bash
# Health check
curl http://localhost:8000/health

# System stats
curl http://localhost:8000/api/stats

# Database status
curl http://localhost:8000/api/health/database
```

## 🚀 Production Deployment

### Docker
```bash
# Build and deploy
docker-compose -f docker-compose.yml up -d

# Scale workers
docker-compose up -d --scale defimon-api=3
```

### PM2
```bash
# Start with PM2
pm2 start run_defimon_v2.py --name defimon-v2

# Monitor
pm2 monit

# Scale
pm2 scale defimon-v2 4
```

## 📊 Производительность

### Оптимизации
- **Индексы**: Оптимизированные индексы для быстрых запросов
- **Кэширование**: Redis для часто используемых данных
- **Партиционирование**: Партиционирование по времени для больших таблиц
- **Асинхронность**: Асинхронная обработка данных

### Масштабирование
- **Горизонтальное**: Множественные API инстансы
- **Вертикальное**: Увеличение ресурсов
- **База данных**: Read реплики для аналитики
- **Кэш**: Redis кластер для высокой доступности

## 🔒 Безопасность

### API Security
- **Аутентификация**: JWT токены
- **Авторизация**: Role-based access control
- **Rate Limiting**: Ограничение запросов
- **Input Validation**: Валидация входных данных

### Database Security
- **Шифрование**: Шифрование соединений
- **Backup**: Регулярные бэкапы
- **Access Control**: Ограничение доступа
- **Audit Logs**: Логирование действий

## 📚 Документация

- [Database Schema](DATABASE_SCHEMA.md) - Подробная схема БД
- [API Documentation](http://localhost:8000/docs) - Swagger UI
- [Architecture](ARCHITECTURE.md) - Архитектура системы
- [Setup Guide](SETUP.md) - Руководство по установке

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📄 Лицензия

MIT License - см. [LICENSE](LICENSE) файл

## 🆘 Поддержка

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discord**: [DEFIMON Community](https://discord.gg/defimon)
- **Email**: support@defimon.com

---

**DEFIMON v2** - Ваш надежный партнер в анализе крипто-активов! 🚀
