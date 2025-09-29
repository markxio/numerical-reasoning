###############################
# Example
###############################

"""
Example:
7UP brands represented an estimated 3.9% of the total domestic soft drink market in 1993, down from 4.0% in 1992.

Task 1: Numerical Entity Extraction

Q: What was the 7UP brands market shrae in 1993?
A: 3.9%

Task 2: Contextual Numerical Reasoning plus multi-hop (maybe across sentences or sections)

Q1: Did 7UP’s market share increase or decrease compared to the previous year?
A1: Decrease
Q2: By how much did the 7UP brands market share decrease?
A2: 0.1% (4.0% - 3.9%)

a) did the value increase or decrease
b) What is the value in the latest year? By how much did the value increase or decrease?

The sentence mentions a metric that decreased or increased from one year to the other year. 
a) Did the metric decrease or increase?
b) By how much did the metric decrease or increase? By how much did the metric change?
"""

#######################################
# Task 1: Numerical Entity Extraction
#######################################

# The sentences were split by nltk.tokenize() and stem from 10-K annual filing documents
# The question is generated with openai's gpt-4o
# The answer is generated with openai's gpt-4o

base="""The following sentence stems from a SEC 10-K annual filing of a company.""" 

generate_qa_instruct = f"""{base} The sentence contains numeric values. Try to understand the setence and then ask a question specifically about the numeric value mentioned in the sentence. Do not generate a question to which the answer is about time, date or years. Instead focus on questions targeted at numeric values such as quantities, percent or money. Then go on, to answer the newly generated question. The answer must be a single numeric value, supplemented by its metric if it is not just a quantity. If you are not certain, then output the following phrase as answer: 'Answer cannot be determined.'

Here is an example with sentence, question and answer:

Example sentence: 7UP rands represented an estimated 3.9% of the total domestic soft drink market in 1993, down from 4.0% in 1992.

Example question: What was the 7UP brands' market share in 1993?

Example answer: 3.9%

Instructions: First, generate the question from the provided sentence, with the question targeted at the numeric value. Second, provide the answer to the question from step 1."""

#######################################
# Task 2: Contextual Numerical Reasoning plus multi-hop
#######################################

# This task uses the same corpus of sentences as Task 1
# i.e., triplets of (sentence, question, answer)
#
# Answers are tagged and labelled by spacy with MONEY and PERCENT
# for both, we have samples with two consecutive years
# and samples with three consecutive years
#
# for MONEY and three consecutive years,
# expect that three amounts of dollars in three years are mentioned in the sentence
# see below for examples 
#
# Note the difference between quantitative and qualitative answers

#######################################
# Task 2: Example answers labelled with MONEY 
#######################################

money_two_consecutive_years_sentence="""
Example sentence: The Company's privately held debt and equity securities and other investments amounted to $4.6 billion and $4.4 billion as of January 31, 2023 and January 31, 2022, respectively.
"""

money_two_consecutive_years_answer_qualitative="""
Example answer: increase
"""

money_two_consecutive_years_answer_quantitative="""
Example answer: $0.2 billion
"""

### 

money_three_consecutive_years_sentence="""
Example sentence: The total compensation cost for all share-based incentive compensation awards recognized in earnings for the years ended December 31, 2012, 2011 and 2010 was $120 million, $121 million and $128 million, respectively.
"""

money_three_consecutive_years_answer_qualitative="""
Example answer: decrease, decrease 
"""

money_three_consecutive_years_answer_quantitative="""
Example answer: $7 million, $1 million
"""

#######################################
# Task 2: Example answers labelled with PERCENT 
#######################################

percent_two_consecutive_years_sentence="""
Example sentence: In addition, to enhance local relevance by having local franchisees operate more restaurants and to improve returns, we reduced our percentage of Company-operated restaurants in the U.K. from 63% at the end of 2005 to 54% at the end of 2006. 
"""

percent_two_consecutive_years_answer_qualitative="""
Example answer: decrease
"""

percent_two_consecutive_years_answer_quantitative="""
Example answer: 9%
"""

percent_three_consecutive_years_sentence="""
Example sentence: Our period-over-period revenue growth was 26% in 2006, 39% in 2007 and 36% for the first three quarters of 2008.
"""

percent_three_consecutive_years_answer_qualitative="""
Example answer: increase, decrease
"""

percent_three_consecutive_years_answer_quantitative="""
Example answer: 13%, 3%
"""

#######################################
# Task 2: Instructions containing examples from above 
#######################################

multi_hop_base = f"""The sentence mentions a financial metric that over the time of two or three years decreased or increased."""

generate_multi_hop_qualitative = f"""{base} {multi_hop_base} Did the financial metric decrease, stay constant or increase from the earlier year to the next later year? For sentences with two years, answer qualitatively with 'increase', 'constant' or 'decrease'. For sentences with three years, answer qualitatively with 'increase, increase', 'increase, decrease', 'increase, constant', 'decrease, decrease', 'decrease, constant', 'decrease, increase', 'constant, increase', 'constant, decrease' or 'constant, constant'.

Here are four examples with sentences and answers:

1. Example

{money_two_consecutive_years_sentence}

{money_two_consecutive_years_answer_qualitative}

2. Example

{percent_two_consecutive_years_sentence}

{percent_two_consecutive_years_answer_qualitative}

3. Example

{money_three_consecutive_years_sentence}

{money_three_consecutive_years_answer_qualitative}

4. Example

{percent_three_consecutive_years_sentence}

{percent_three_consecutive_years_answer_qualitative}
"""

generate_multi_hop_quantitative = f"""{base} {multi_hop_base} By how much did the financial metric change from the earlier year to the next later year? For sentences with two years, answer quantitatively by stating the numeric difference. For sentences with three years, answer quantitatively by stating the numeric difference between two subsequent years.

Here are four examples with sentences and answers:

1. Example

{money_two_consecutive_years_sentence}

{money_two_consecutive_years_answer_quantitative}

2. Example

{percent_two_consecutive_years_sentence}

{percent_two_consecutive_years_answer_quantitative}

3. Example

{money_three_consecutive_years_sentence}

{money_three_consecutive_years_answer_quantitative}

4. Example

{percent_three_consecutive_years_sentence}

{percent_three_consecutive_years_answer_quantitative}
"""
