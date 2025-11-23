#!/bin/bash
set -e
PYTHONPATH=. python src/services/predict_company/train_all_products.py
PYTHONPATH=. python src/services/predict_company/train_all_companies.py
python src/services/predict_all_tickets/train_all_tickets.py
python controller/tickets_controller.py
