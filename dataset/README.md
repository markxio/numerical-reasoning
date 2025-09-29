# Dataset

Columns:
- **json_file_path**: path to the file created by *edgar-crawler*
- **year**: report year
- **ticker_id**: tickers are unique identifiers for corporate identities in the SEC's EDGAR database
- **item**: essentially sections (reports are structured in numbered items)
- **sentence_id**: we split each item (section) into sentences, assigning incremental identifiers per sentence
- **answer_labels**: spacy labels (MONEY or PERCENT)
- **context**: For the long context, the **context** column contains the whole item. For the short context, the **context** column contains the individual sentence
- **task_3_question**: context-specific question probing the value of the financial metric in year0
- **task_4_question**: context-specific question probing the qualitative change of the financial metric from year0 to year1
- **task_5_question**: context-specific question probing the quantitative change in value of the financial metric from year0 to year1
- **task_1_answer**: the financial metric reported in the short context
- **task_2_answer**: a comma-separated list of years that the financial metric is reported for 
- **task_3_answer**: the financial metric's value in year0
- **task_4_answer**: [increase, decrease, constant]
- **task_5_answer**: the difference between [financial metric in year1] - [financial metric in year0]
