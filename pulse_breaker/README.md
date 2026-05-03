# Pulse Breaker — Backtest Project

## Folder Structure

```
pulse_breaker/
├── ideas.txt                        ← All experiment ideas + version history
├── README.md                        ← This file
│
├── weekly/
│   ├── scripts/
│   │   ├── v1/  02_strategy_v1.py  ← Baseline (RS>0, 2% trail)
│   │   ├── v2/  02_strategy_v2.py  ← (next)
│   │   └── v3/  ...
│   └── results/
│       ├── v1/  trade_log.csv  summary.csv
│       ├── v2/  ...
│       └── v3/  ...
│
└── daily/                           ← Future
    ├── scripts/v1/
    └── results/v1/
```

## Data Source
All scripts pull from:
`backtesting/common/data/weekly/`

## How to Run
```bash
# Always run from the script's own folder
cd pulse_breaker/weekly/scripts/v1
python 02_strategy_v1.py

# Results saved automatically to:
# pulse_breaker/weekly/results/v1/trade_log.csv
# pulse_breaker/weekly/results/v1/summary.csv
```

## Version Log
| Version | Key Change | Win Rate | Total P&L |
|---------|-----------|----------|-----------|
| v1      | Baseline  | 56.5%    | Rs 98,512 |