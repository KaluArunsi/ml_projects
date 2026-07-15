# ml_projects

a collection of machine learning and ai projects i build and ship. each project tackles a different problem space, from structured data to nlp to full-stack apps.

## projects

**[oil spill cause classification](oil-spill-cause-classification/)**
multi-label nlp system that classifies the cause of oil spills from incident descriptions and social media posts. three modeling tracks: tf-idf baseline, distilbert fine-tuning with focal loss, and a qwen2.5 model fine-tuned via mlx lora on apple silicon. includes a streamlit app for training, prediction, and reporting.

**[credit worthiness assessment](credit-worthiness/)**
probabilistic loan default classifier built on lendingclub application data. estimates the likelihood a borrower defaults and surfaces the key features driving that prediction. every output is a complementary probability pair.

**[hotel cancellation risk](hotel-cancellation-determination/)**
two-model system that predicts booking cancellations and assigns severity tiers. built so a revenue team can look at a dashboard and know which bookings need attention first, without digging through spreadsheets.

**[bike demand forecasting](bike-demand-prediction/)**
hourly bike demand prediction for seoul's public bike sharing system. practical time-series modeling with weather, seasonal, and calendar features.

**[openbpo drift](openbpo-drift/)**
a monitoring and observability tool for tracking drift in outsourced business process operations. built with streamlit.

## structure

every project folder is self-contained with its own dependencies, data pipeline, and documentation. most projects include a streamlit interface or jupyter notebook for exploration, plus a cli entry point for scripting and automation.

## tech

python across the board. the stack varies by project but pulls from: scikit-learn, pytorch, transformers, mlx (apple silicon), streamlit, plotly, pandas, numpy, and the usual data science toolchain.
